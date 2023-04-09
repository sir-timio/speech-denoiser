import os
from webbrowser import get
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import sys
sys.path.append('../wave_wizard')

import torch
import pytorch_lightning as pl
from src.dataset import get_loader
from src.gatewave import LitGateWave
from src.blocks import *
from src.dataset import get_loader
from clearml import Task
from clearml.automation import PipelineController

task = Task.init(project_name="examples", task_name="Pipeline step 2 train model")
args = {
    'data_loader': '',
}
task.connect(args)
task.execute_remotely()
data_loader_upload_task = Task.get_task(task_id=args['data_loader'])
# data_loader = data_loader_upload_task.artifacts['data_loader']

model_config = dict(
    depth=3, scale=2, init_hidden=32,
    kernel_size=7, stride=1, padding=2,
    encoder_class=BasicConv,
    decoder_class=BasicDeConv
)
model = LitGateWave(**model_config)
task._add_artifacts(model_config)
trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else None,
    devices=1,
    check_val_every_n_epoch=1,
    log_every_n_steps=50,
    deterministic=True,
)
data_config_upload_task = Task.get_task(task_id=args['dataset_task_id'])
data_config = data_config_upload_task.artifacts['data_config']
task.connect(model_config)
data_loader = get_loader(data_config)
trainer.fit(model, train_dataloaders=data_loader)