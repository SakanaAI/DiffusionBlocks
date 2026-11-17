import datetime
import os
from dataclasses import replace
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from config import ModelType, RunConfig, Stage, parse_args
from data import ImageDataModule, load_data
from evaluation import SampleGrid, SaveResults
from model import load_model

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def test_callbacks(
    args: RunConfig, data: ImageDataModule, logdir: str
) -> list[L.Callback]:
    """Evaluation run in the test stage, beyond what the model logs itself."""
    match args.model_type:
        case ModelType.DIT:
            # per-seed names, so evaluations don't overwrite each other
            return [
                SampleGrid(Path(logdir) / "grid.png", data.num_labels),
                SaveResults(
                    Path(logdir) / f"results_seed{args.seed}.json", args.to_dict()
                ),
            ]
        case _:
            return []


def main(args: RunConfig) -> None:
    L.seed_everything(args.seed)

    data = load_data(args)
    args = replace(args, image_size=data.image_size, num_labels=data.num_labels)
    model = load_model(args)
    if args.ckpt_path is not None:
        nowname = os.path.basename(os.path.dirname(args.ckpt_path))
    else:
        now = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9), name="JST")
        ).strftime("%Y-%m-%dT%H-%M-%S")
        nowname = now + f"-{args.model_type}" + args.postfix
        nowname = nowname.removeprefix("_")
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
        # to align total number of iterations across the entire network because one step corresponds to one block
        max_epochs=args.num_epochs * model.num_blocks,
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
            *test_callbacks(args, data, logdir),
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        strategy=DDPStrategy(find_unused_parameters=model.num_blocks > 1)
        if args.devices > 1
        else "auto",
        devices=args.devices,
        logger=logger,
        num_sanity_val_steps=0,
        # precision="bf16-mixed",
    )
    if args.stage == Stage.TRAIN:
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
        if data.test_key is not None:
            trainer.test(model, data.test_dataloader(), ckpt_path="best")
    else:
        assert args.ckpt_path is not None
        trainer.test(model, data, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main(parse_args())
