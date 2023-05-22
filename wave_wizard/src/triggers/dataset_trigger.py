from clearml import Task
from clearml.automation import TriggerScheduler

task = Task.init(
    project_name="DevOps",
    task_name="audio_dataset_update",
)

trigger = TriggerScheduler(pooling_frequency_minutes=0.5)
TASK_ID = "df7b88be17654a6d9b0ee948cac6429f"
trigger.add_dataset_trigger(
    schedule_queue="default",
    name="fit on updated dataset",
    schedule_task_id=TASK_ID,
    trigger_project="GateWave",
)

trigger.start_remotely(queue="default")

# get parent from clearml info
# clearml-data sync --project SpeechDenoiseData --name SpeechDenoiseData --parent c45f8e1ef5f44123b3dc2a6bcfbe22ca --folder /mnt/data/MS-SNSD/
