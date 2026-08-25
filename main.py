import os
import argparse
import datetime
import json
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy

from data import load_data
from model import load_model

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    L.seed_everything(args.seed)

    data = load_data(args)
    args.image_size = data.image_size
    args.num_labels = data.num_labels
    model = load_model(args)
    if args.ckpt_path is not None:
        nowname = os.path.basename(os.path.dirname(args.ckpt_path))
    else:
        now = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9), name="JST")
        ).strftime("%Y-%m-%dT%H-%M-%S")
        nowname = now + f"-{args.model_type}" + args.postfix
        if nowname.startswith("_"):
            nowname = nowname[1:]
    print("Experiment Name:", nowname)
    logdir = os.path.join("logs", nowname)
    logger = WandbLogger(
        project=f"dblocks-{args.data_name}",
        name=nowname,
        version=nowname,
        offline=args.debug,
        save_dir=logdir,
        # group=f"{args.data_name}",
    )
    trainer = L.Trainer(
        max_epochs=args.num_epochs
        if args.model_type != "dblock"
        else args.num_epochs
        * args.num_blocks,  # to align total number of iterations across the entire network because one step corresponds to one block
        check_val_every_n_epoch=args.save_every_n_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=logdir,
                monitor="val/acc" if data.val_key is not None else None,
                mode="max",
                save_top_k=args.save_top_k,
                save_on_train_epoch_end=True,
                every_n_epochs=args.save_every_n_epochs
                if data.val_key is None
                else None,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        strategy=DDPStrategy(find_unused_parameters=args.model_type == "dblock")
        if args.devices > 1
        else "auto",
        devices=args.devices,
        logger=logger,
        num_sanity_val_steps=0,
        # precision="bf16-mixed",
    )
    if args.stage == "train":
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
        if data.test_key is not None:
            # "best" is only populated when a metric is monitored. Datasets
            # without a validation split (cifar100) leave monitor=None, so a run
            # shorter than save_every_n_epochs never writes a "best" checkpoint.
            # save_last=True is always set, so "last" is available either way.
            ckpt = "best" if data.val_key is not None else "last"
            trainer.test(model, data.test_dataloader(), ckpt_path=ckpt)
    else:
        assert args.ckpt_path is not None
        trainer.test(model, data, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, default="train", choices=["train", "test"])
    parser.add_argument("data_name", type=str, default="cifar100")
    parser.add_argument(
        "--model_type", type=str, default="vit", choices=["vit", "dblock"]
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--add_rand_aug", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument(
        "--scheduler_specific_kwargs",
        type=json.loads,
        default=None,
        help="specific kwargs for the scheduler",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--deepspeed", action="store_true", help="use deepspeed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--postfix", type=str, default="", help="postfix for the experiment name"
    )
    # dblock
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--class_dropout_prob", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
