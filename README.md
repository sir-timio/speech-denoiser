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
run training, check metrics and debug samples

```
    python train.py --config configs/train_config.yaml
```
