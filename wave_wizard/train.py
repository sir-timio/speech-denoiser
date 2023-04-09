import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import pytorch_lightning as pl
from src.blocks import BasicConv, BasicDeConv, GatedConv, GatedDeConv
from src.dataset import get_loader
from src.gatewave import LitGateWave
from clearml import Task
from clearml.automation import PipelineController
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/config.yaml',
            help='YAML configuration file')
    return parser.parse_args()

def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':
    pl.seed_everything(42)
    config = load_config(parse_args())
    task = Task.init(project_name="GateWave", task_name="Example")
    
    print(config)

    train_loader = get_loader(config['dataset'])
    val_loader = get_loader(config['dataset'])

    model_config = dict(
        depth=3, scale=2, init_hidden=32,
        kernel_size=7, stride=1, padding=2,
        encoder_class=BasicConv,
        decoder_class=BasicDeConv
    )

    model = LitGateWave(**model_config)
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else None,
        devices=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        deterministic=True,
    )
    task.connect(model_config)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)