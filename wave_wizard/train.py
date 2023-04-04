import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import pytorch_lightning as pl
from src.blocks import BasicConv, BasicDeConv, GatedConv, GatedDeConv
from src.dataset import get_loader
from src.gatewave import LitGateWave
from clearml import Task
from clearml.automation import PipelineController


if __name__ == '__main__':
    pl.seed_everything(42)
    
    pipe = PipelineController(
        name="Pipeline demo", project="examples", version="0.0.1", add_pipeline_tags=False
    )
    # task = Task.init(project_name="GateWave", task_name="Example")
    SR = 22050
    SECS = 11
    length=SR*SECS
    sample_rate=22050
    data_config = dict({
        'dataset': {
            'json_dir': 'dataset',
            'length': SR*SECS,
            'sample_rate': SR,
            'num_samples': 10_000,
        },
        'dataloader': {
            'batch_size': 32,
            'num_workers': 10,
        }
    })
    
    pipe.add_parameter(
        "data_config",
        data_config,
    )
    
    pipe.set_default_execution_queue('default')
    
    pipe.add_step(
        name="stage_data",
        base_task_project="examples",
        base_task_name="Pipeline step 1 dataset artifact",
        parameter_override={"General/data_config": "${pipeline.data_config}"},
    )
    
    # train_loader = get_loader(data_config)
    # val_loader = get_loader(data_config)

    model_config = dict(
        depth=3, scale=2, init_hidden=32,
        kernel_size=7, stride=1, padding=2,
        encoder_class=BasicConv,
        decoder_class=BasicDeConv
    )
    pipe.add_parameter(
        "model_config",
        model_config,
    )
    # model = LitGateWave(**model_config)
    
    pipe.add_step(
        name="stage_train",
        parents=["stage_data"],
        base_task_project="examples",
        base_task_name="Pipeline step 2 train model",
        parameter_override={
            "General/data_loader": "${stage_data.data_loader}",
        },
    )
    # trainer = pl.Trainer(
    #     accelerator='gpu' if torch.cuda.is_available() else None,
    #     devices=1,
    #     check_val_every_n_epoch=1,
    #     log_every_n_steps=50,
    #     deterministic=True,
    # )
    
    # task.connect(model_config)
    
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.fit(model, train_dataloaders=data_loader)