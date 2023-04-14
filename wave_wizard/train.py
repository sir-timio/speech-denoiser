import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from src.dataset import get_train_val_test_loaders
from clearml import Task
import argparse
import yaml
from omegaconf import DictConfig

from src.loss import MultiResolutionSTFTLoss
# from src.models import Demucs, GateWave, WaveUnet
from src.models.demucs import Demucs
from src.models.gatewave import GateWave
from src.models.unet import WaveUnet
from pytorch_lightning.callbacks import ModelCheckpoint
from src.callbacks import AudioVisualizationCallback, MetricCallback
from torch.utils.tensorboard import SummaryWriter
from src.wrap import LitModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/train_config.yaml',
            help='YAML configuration file')
    return parser.parse_args()

def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_model(config):
    name = config['model']['type']
    if name == 'gatewave':
        cls = GateWave
    elif name == 'demucs':
        cls = Demucs
    elif name == 'waveunet':
        cls = WaveUnet
    else:
        raise Exception("Unknown architecture")

    return cls(**config['model']['args'])

if __name__ == '__main__':
    config = load_config(parse_args())
    task = Task.init(project_name="GateWave", task_name=config['model']['type'])
    logger = task.get_logger()
    writer = SummaryWriter(config['trainer']['log_dir'])
    pl.seed_everything(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
    mrstftloss = MultiResolutionSTFTLoss(
        factor_sc=config.loss.stft_sc_factor,
        factor_mag=config.loss.stft_mag_factor,
    )
    
    def loss_fn(x, y):
        sc_loss, mag_loss = mrstftloss(x.squeeze(1), y.squeeze(1))
        return F.l1_loss(x, y) + sc_loss + mag_loss
    
    train_loader, val_loader, test_loader = get_train_val_test_loaders(config['dataset'])

    vis_callback = AudioVisualizationCallback(
        test_loader,
        writer,
        every_n_epochs=config['trainer']['debug_interval'],
        )
    metric_callback = MetricCallback(test_loader, config['trainer']['metric_interval'])
    model = get_model(config)
    
    pl_model = LitModel(config, model, loss_fn, logger)
    
    checkpoint_path = os.path.join(
        config.trainer.base_dir, config.trainer.exp_name, "checkpoints"
    )
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best-checkpoint",
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )
    trainer = pl.Trainer(
        accelerator='gpu' if device == 'cuda' else None,
        max_epochs=config['trainer']['epochs'],
        devices=1,
        check_val_every_n_epoch=config['trainer']['val_interval'],
        log_every_n_steps=config['trainer']['log_steps'],
        deterministic=False,
        callbacks=[vis_callback, checkpoint_callback, metric_callback],
    )
    task.connect(config)
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)