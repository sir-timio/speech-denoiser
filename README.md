# Web speech denoiser and transcriber

python 3.9

# Структура проекта:
```
.
├── app # web and live applications
├── doc # some samples, license, presentation
└── wave_wizard # modeling, fitting, test and so on.
```

[Результаты обучения](wave_wizard/README.md)

[Запуск приложений](app/README.md)

### Инструкция по запуску:
```
git clone https://github.com/sir-timio/web-denoiser.git
cd web-denoiser
pip install -r full_requirements.txt 

```
download/get clean and noise audio files. Used data: https://github.com/microsoft/MS-SNSD:

```
    git clone https://github.com/microsoft/MS-SNSD
    mv MS-SNSD/noise_train wave_wizard/noise_train
    mv MS-SNSD/noise_train wave_wizard/clean_train
```


optional: add to clean_data some subsets from https://github.com/snakers4/open_stt/#links with 99%+ quality


generate train data, split it into train, val and test as you wish, but test metrics take a lot of time

```
    cd wave_wizard
    python noiser.py --config configs/noiser_config.yaml
```

basic training:
run, check metrics and debug samples

```
    python train.py --config configs/train_config.yaml
```

advanced training with clearml queue and triggers:

run first experiment to get ID for src/triggers/dataset_trigger.py
```
    python train.py --config configs/train_config.yaml
```

add dataset with [clearml-data](https://clear.ml/docs/latest/docs/clearml_data/), example:
```
    clearml-data create --project<project_name> --name <dataset_name>
    clearml-data add --files <path_to_data>
```

configure src/triggers/dataset_trigger.py

```
clearml-data sync --project <project_name> --name <dataset_name> --parent <parent_dataset_id> --folder <path_to_new_data>
```

I recommend name child dataset as parent, thus you could get newest version of dataset by name.

and finally, run task, manage queue with web.
```
    python src/triggers/dataset_trigger.py
```

As soon as dataset would be updated, task would automatically started again.