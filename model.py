import random
import numpy as np
from scipy.stats import norm
import torch
import torch.nn.functional as F
import lightning as L
import torchmetrics
from transformers import get_scheduler

from vit import load_vit
from dblock_modules import get_block_sigmas, get_discrete_sigmas


def load_model(args):
    if args.model_type == "vit":
        return ViTModel(args)
    elif args.model_type == "dblock":
        return ViTDBlockModel(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")


class ViTModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = args.image_size
        self.num_labels = args.num_labels
        self.valid_metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.num_labels
                ),
                "f1": torchmetrics.F1Score(
                    task="multiclass", num_classes=self.num_labels
                ),
            },
            prefix="val/",
        )
        self.test_metrics = self.valid_metrics.clone(prefix="test/")
        self.save_hyperparameters(args)

    def configure_model(self):
        self.model = load_vit(image_size=self.image_size, num_labels=self.num_labels)
        print(self.model)
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_scheduler(
            name=self.args.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            scheduler_specific_kwargs=self.args.scheduler_specific_kwargs,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, **kwargs):
        return self.model(**kwargs).logits

    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values=pixel_values, **kwargs)
        if return_metrics:
            if step == "val":
                return self.valid_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif step == "test":
                return self.test_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            else:
                raise NotImplementedError(f"Step {step} is not supported")

        loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        loss_dict = {f"{step}/loss": loss}
        return loss, loss_dict

    def get_model_kwargs(self, batch):
        return {}

    def training_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        loss, loss_dict = self.shared_step(batch, step="train", **model_kwargs)
        self.log_dict(loss_dict, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(batch, step="val", return_metrics=True, **model_kwargs)
        self.log_dict(res, batch_size=batch_size, prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(batch, step="test", return_metrics=True, **model_kwargs)
        self.log_dict(res, batch_size=batch_size, prog_bar=True)


class ViTDBlockModel(ViTModel):
    def __init__(self, args):
        super().__init__(args)
        self.gamma = args.gamma
        self.sigma_data = 0.5
        self.cfg_scale = args.cfg_scale
        self.class_dropout_prob = (
            args.class_dropout_prob if self.cfg_scale > 0.0 else 0.0
        )
        self.num_inference_steps = self.args.num_inference_steps or self.args.num_blocks
        self.block_sigmas = get_block_sigmas(num_layers=self.args.num_blocks)
        self.layer_assignment = None
        self.register_buffer(
            "sigmas",
            get_discrete_sigmas(num_steps=self.num_inference_steps, dblock=True).to(
                self.device
            ),
        )
        self.save_hyperparameters(
            {
                "gamma": self.gamma,
                "num_inference_steps": self.num_inference_steps,
                "cfg_scale": self.cfg_scale,
                "class_dropout_prob": self.class_dropout_prob,
            },
        )

    def configure_model(self):
        self.model = load_vit(
            image_size=self.image_size, num_labels=self.num_labels, is_dblock=True
        )
        print(self.model)

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def get_embeds(
        self, input_ids: torch.Tensor, is_input: bool = True
    ) -> torch.Tensor:
        if is_input:
            embeds = self.model.get_input_embeddings()(input_ids)
        else:
            embeds = F.embedding(
                input_ids, weight=self.model.get_output_embeddings().weight
            )
        return self.normalize_embeddings(embeds)

    def get_sigmas(self, n_samples: int, p_mean: float = -1.2, p_std: float = 1.2):
        block_idx = random.choices(range(self.args.num_blocks), k=1)[0]
        sigma_min_block = self.block_sigmas[block_idx]
        sigma_max_block = self.block_sigmas[block_idx + 1]
        # extend the range
        if self.gamma > 0.0:
            log_sigma_min = np.log(sigma_min_block)
            log_sigma_max = np.log(sigma_max_block)
            log_range = log_sigma_max - log_sigma_min
            sigma_min_block = np.exp(log_sigma_min - self.gamma * log_range)
            sigma_max_block = np.exp(log_sigma_max + self.gamma * log_range)
            sigma_min_block = max(sigma_min_block, self.block_sigmas[0])
            sigma_max_block = min(sigma_max_block, self.block_sigmas[-1])

        cdf_min_block = norm.cdf((np.log(sigma_min_block) - p_mean) / p_std)
        cdf_max_block = norm.cdf((np.log(sigma_max_block) - p_mean) / p_std)

        rand = np.random.uniform(cdf_min_block, cdf_max_block, n_samples)
        sigma = np.exp(p_mean + p_std * norm.ppf(rand))
        sigma = torch.from_numpy(sigma)
        return sigma

    def get_weights(self, sigmas):
        return (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data) ** 2

    def estimate_target_layer(self, sigma: torch.Tensor) -> int:
        block_sigmas = torch.tensor(self.block_sigmas, device=sigma.device)
        block_idx = torch.bucketize(sigma, block_sigmas, right=True) - 1
        block_idx = (self.args.num_blocks - 1) - block_idx
        block_idx = torch.clamp(block_idx, 0, self.args.num_blocks - 1).long()
        values, counts = block_idx.unique(return_counts=True)
        return values[counts.argmax()].item()

    def denoise(self, x, zt, sigma, block_idx=None):
        if block_idx is None:
            block_idx = self.estimate_target_layer(sigma)
        if self.class_dropout_prob > 0.0 and self.training:
            drop_x = torch.rand(x.shape[0], device=x.device) < self.class_dropout_prob
            uncond_x = torch.zeros_like(x)
            x = torch.where(drop_x[:, None, None, None], uncond_x, x)
        elif not self.training and self.cfg_scale > 0.0:
            uncond_x = torch.zeros_like(x)
            x = torch.cat([uncond_x, x])
            zt = torch.cat([zt] * 2)
            sigma = torch.cat([sigma] * 2)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()

        if self.layer_assignment is None:
            split_size = self.model.config.num_hidden_layers // self.args.num_blocks
            self.layer_assignment = [
                list(range(i * split_size, (i + 1) * split_size))
                for i in range(self.args.num_blocks)
            ]
        outputs = self.model.forward_block(
            layer_indices=self.layer_assignment[block_idx],
            pixel_values=x,
            noisy_embeds=zt * c_in[:, None],
            timesteps=c_noise,
        )
        hidden_states = outputs.last_hidden_state
        conditioning = outputs.conditioning
        model_out = hidden_states * c_out[:, None] + zt * c_skip[:, None]
        logits = self.model.forward_output_embeddings(
            model_out.unsqueeze(1), conditioning
        )
        if not self.training and self.cfg_scale > 0.0:
            logits_uncond, logits_cond = logits.chunk(2)
            logits = logits_uncond + self.cfg_scale * (logits_cond - logits_uncond)
        return logits

    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        if return_metrics:
            logits = self.diffusion_step(pixel_values)
            if step == "val":
                return self.valid_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif step == "test":
                return self.test_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            else:
                raise NotImplementedError(f"Step {step} is not supported")

        z = self.get_embeds(labels, is_input=True)
        sigmas = self.get_sigmas(z.shape[0])
        block_idx = self.estimate_target_layer(sigmas)
        sigmas = sigmas.to(z)
        zt = z + sigmas[:, None] * torch.randn_like(z)
        logits = self.denoise(pixel_values, zt, sigmas, block_idx)
        loss = F.cross_entropy(
            logits.view(-1, self.num_labels), labels.view(-1), reduction="none"
        )
        ce_loss = loss.mean()
        w = self.get_weights(sigmas)[:, None]
        loss = (loss * w).mean()

        loss_dict = {
            f"{step}/loss": loss,
            f"{step}/loss_{block_idx}": loss,
            f"{step}/ce_loss": ce_loss,
            f"{step}/ce_loss_{block_idx}": ce_loss,
        }
        return loss, loss_dict

    def diffusion_step(self, x):
        bsz = x.shape[0]
        hidden_size = self.model.config.hidden_size
        z = torch.randn(bsz, hidden_size, device=self.device)
        z *= torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = x.new_ones([x.shape[0]])
        for i in range(self.sigmas.shape[0] - 1):
            sigma = self.sigmas[i] * s_in
            next_sigma = self.sigmas[i + 1] * s_in
            # denoise
            logits = self.denoise(x, z, sigma)
            probs = F.softmax(logits, dim=1)
            denoised = F.linear(probs, self.model.get_input_embeddings().weight.t())
            # to d
            d = (z - denoised) / sigma[:, None]
            dt = next_sigma - sigma
            # Descending sigma means dt <= 0. Equivalently, add
            # ((sigma - next_sigma) / sigma) * (denoised - z).
            euler_step = z + dt[:, None] * d
            z = euler_step
        min_sigma = self.sigmas[-1].item()
        sigmas = torch.full((x.shape[0],), min_sigma, device=self.device)
        logits = self.denoise(x, z, sigmas)
        return logits

