import os
from typing import Callable

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import functools

import optuna
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from addict import Dict
from clearml import Task
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter

from src.callbacks import AudioVisualizationCallback, MetricCallback
from src.dataset import get_train_val_test_loaders
from src.loss import MultiResolutionSTFTLoss

# from src.models import Demucs, GateWave, WaveUnet
from src.models.demucs import Demucs
from src.models.gatewave import GateWave
from src.models.waveunet import WaveUnet
from src.wrap import LitModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="?",
        default="configs/train_config.yaml",
        help="YAML configuration file",
    )
    return parser.parse_args()


def load_config(args) -> dict:
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return Dict(config)


def get_loss_fn(config) -> Callable:
    """
    build loss fn based on config.

    Args:
        config (dict): exp config

    Raises:
        Exception: One of losses should have non zero weight

    Returns:
        Callable: loss
    """
    if config.loss.stft_weight != 0:
        mrstftloss = MultiResolutionSTFTLoss(
            factor_sc=config.loss.stft_sc_factor,
            factor_mag=config.loss.stft_mag_factor,
        )
    if config.loss.l1_weight != 0 and config.loss.stft_weight != 0:
        alpha = config.loss.l1_weight
        beta = config.loss.stft_weight
        return lambda x, y: dict(
            l1=F.l1_loss(x, y) * alpha,
            mstft=sum(mrstftloss(x.squeeze(1), y.squeeze(1))) * beta,
        )
    elif config.loss.l1_weight != 0 and config.loss.stft_weight == 0:
        return lambda x, y: dict(l1=F.l1_loss(x, y) * config.loss.l1_weight)
    elif config.loss.l1_weight == 0 and config.loss.stft_weight != 0:
        return lambda x, y: dict(
            mstft=sum(mrstftloss(x.squeeze(1), y.squeeze(1))) * config.loss.stft_weight
        )
    else:
        raise Exception("One of losses should have non zero weight")


def get_model(config) -> torch.nn.Module:
    """
    build model based on config

    Args:
        config (dict): exp config

    Raises:
        Exception: Passed unknown arch

    Returns:
        torch.nn.Module: model instance
    """
    name = config.model.type
    if name == "gatewave":
        cls = GateWave
    elif name == "demucs":
        cls = Demucs
    elif name == "waveunet":
        cls = WaveUnet
    else:
        raise Exception("Unknown architecture")

    return cls(**config.model.args)


def train_model(trial, config):
    config.loss.l1_weight = trial.suggest_int("l1_weight", 0, 10)
    config.loss.stft_weight = trial.suggest_int("stft_weight", 0, 10)
    if config.loss.stft_weight == 0 and config.loss.l1_weight == 0:
        config.loss.stft_weight = 1
    task_name = (
        f"{config.model.type}_l1:{config.loss.l1_weight}_stft:{config.loss.stft_weight}"
    )
    config.trainer.exp_name = task_name
    task = Task.init(project_name="GateWave", task_name=task_name)
    writer = SummaryWriter(config.trainer.log_dir)
    pl.seed_everything(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = get_train_val_test_loaders(config.dataset)

    vis_callback = AudioVisualizationCallback(
        test_loader,
        writer,
        every_n_epochs=config.trainer.debug_interval,
    )
    metric_callback = MetricCallback(test_loader, config.trainer.metric_interval)
    model = get_model(config)
    loss_fn = get_loss_fn(config)

    pl_model = LitModel(config, model, loss_fn)
    checkpoint_path = os.path.join(
        config.trainer.base_dir, config.trainer.exp_name, "checkpoints"
    )
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=30,
        dirpath=checkpoint_path,
        filename="best-checkpoint",
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else None,
        max_epochs=config.trainer.epochs,
        devices=1,
        check_val_every_n_epoch=config.trainer.val_interval,
        log_every_n_steps=config.trainer.log_steps,
        deterministic=False,
        callbacks=[vis_callback, checkpoint_callback, metric_callback],
    )
    task.connect(config)
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    metrics = trainer.test(pl_model, test_loader)[0]
    task.close()
    return metrics["pesq"]


if __name__ == "__main__":
    config = load_config(parse_args())
    objective = functools.partial(train_model, config=config)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=1000, timeout=60 * 60 * 24)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
