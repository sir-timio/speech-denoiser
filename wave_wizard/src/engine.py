# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# import torch
# import pytorch_lightning as pl
# from src.blocks import BasicConv, BasicDeConv, GatedConv, GatedDeConv
# from src.dataset import get_loader
# from src.gatewave import LitGateWave
# from clearml import Task
# from clearml.automation import PipelineController

# def train_model():
#     pl.seed_everything(42)
        
#     trainer = pl.Trainer(
#         accelerator='gpu' if torch.cuda.is_available() else None,
#         devices=1,
#         check_val_every_n_epoch=1,
#         log_every_n_steps=50,
#         deterministic=True,
#     )
    
#     task.connect(model_config)
    
#     # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
#     # trainer.fit(model, train_dataloaders=data_loader)