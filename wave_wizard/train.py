import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from src.dataset import get_train_val_loaders
from clearml import Task
import argparse
import yaml
from omegaconf import DictConfig

from src.loss import MultiResolutionSTFTLoss
# from src.models import Demucs, GateWave, WaveUnet
from src.models.demucs import Demucs
from src.models.gatewave import GateWave
from src.models.unet import WaveUnet
from src.wrap import LitModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/config.yaml',
            help='YAML configuration file')
    return parser.parse_args()

def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return DictConfig(config)

def get_model(name, config):
    if name == 'gatewave':
        return GateWave(**config['gatewave'])
    elif name == 'demucs':
        return Demucs(**config['demucs'])
    elif name == 'waveunet':
        return WaveUnet()
    raise Exception("Unknown architecture")

if __name__ == '__main__':
    config = load_config(parse_args())
    MODEL_NAME = 'gatewave'
    task = Task.init(project_name="GateWave", task_name=MODEL_NAME)
    pl.seed_everything(config.seed)    
    
    # loss_fn = torch.nn.MSELoss()
    
    mrstftloss = MultiResolutionSTFTLoss(
        factor_sc=config.loss.stft_sc_factor,
        factor_mag=config.loss.stft_mag_factor,
    )
    def loss_fn(x, y):
        sc_loss, mag_loss = mrstftloss(x.squeeze(1), y.squeeze(1))
        return F.l1_loss(x, y) + sc_loss + mag_loss
    
    train_loader, val_loader = get_train_val_loaders(config['dataset'])

    model = get_model(MODEL_NAME, config)
    
    pl_model = LitModel(config, model, loss_fn, task.get_logger())
    
    
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else None,
        devices=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        deterministic=False,
    )
    task.connect(config)
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)