import sys
sys.path.append('../wave_wizard')
import numpy as np
from torch.utils.data import Subset, DataLoader
from src.dataset import NoisyCleanSet
from clearml import Task, StorageManager

task = Task.init(project_name="examples", task_name="Pipeline step 1 dataset artifact")
args = {
    'config': '',
}
task.connect(args)
task.execute_remotely()
data_config_upload_task = Task.get_task(task_id=args['dataset_task_id'])
config = data_config_upload_task.artifacts['data_config']
num_samples = config['dataset'].pop('num_samples', None)
dataset = NoisyCleanSet(**config['dataset'])
idx = np.arange(len(dataset))
if num_samples is not None:
    idx = np.random.choice(idx, num_samples)
    dataset = Subset(dataset, idx)

data_loader = DataLoader(dataset, **config['dataloader'])
task.upload_artifact('data_loader', artifact_object=data_loader)