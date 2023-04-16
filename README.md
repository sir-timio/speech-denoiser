# Web speech denoiser and transcriber


<!-- краткий отчет о проведенных экспериментах
код обучения моделей
обоснование выбора моделей и гиперпараметров
описание кода обучения -->

### Краткий отчет о проведенных эксприментах:

Реализация моделей, цикла обучения представлена в директории wave_wizard.
Было реализованно 3 варианта архитектуры: Wave-Unet и [Demucs](https://github.com/facebookresearch/denoiser) взято практически без изменений, Wave-Unet адаптированно со [статьи](https://arxiv.org/pdf/1806.03185.pdf) и  GateWave - собственная разработка, вдохновленная [статьей](https://paperswithcode.com/method/gated-convolution-network). (Позднее оказалось, что данная идея реализована в функции активации demucs при параметре [glu](https://pytorch.org/docs/stable/generated/torch.nn.GLU.html) (Gated Linear Unit), что объясняет четный рост размерности: вторая половина матрицы представляет из себя 'маску')


В качестве фреймворка для трекинга экспериментов был выбран [CleaML](https://clear.ml/) в связке с [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html). Он удовлетворяет всем необходимым потребностям для данного эксперимента: треккинг метрик, дебаг вывод аудио и изображений, вывод загруженности гпу и процессора, совместимость с pytorch_lightning. Модели реализованы на [Pytorch](https://pytorch.org/), цикл обучения на [Pytorch_Lightning](https://www.pytorchlightning.ai/index.html), рассчет метрик с помощью [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)



Гиперпараметры взяты из статей, для GateWave подбирались по аналогии с WaveUnet и Demucs.

Все модели удовлетворяют запросу - быстрая работа на CPU, поэтому, исходя из результатов обучения, выбор пал на Demucs 64.

Краткое описание цикла обучения:
 - Загрузка чистых аудиозаписей и шума;
 - Комбинация чистых аудиозаписей и шума с различным соотношением, аугментации;
 - Старт обучения модели;
 - Логгирование лосса, вывод изображений и обратанных аудио раз в n эпох, рассчет метрик (stoi, pesq, snr).
 - Сохранение модели для инференса


### Структура проекта:
```
.
├── app # веб приложение
├── data # кастомные шумы, хранилище веб приложения
├── doc # оформление репозитория
├── environment.yml
├── README.md
└── wave_wizard # модели, обучение, тестирование и т.д. 
```

### Структура wave_wizard:
```
.
├── blocks # реализация кастомных блоков для GateWave
│   ├── basic.py
│   └── gated.py
├── callbacks.py # pytorch_ligthing callbacks для визуализации процесса обучения (дебаг аудио, изображений)
├── dataset.py # обработка данных, dataloaders
├── inference.py # инференс модели
├── loss.py # stft лосс из demucs
├── models # реализация моделей, описанных выше
│   ├── demucs.py
│   ├── gatewave.py 
│   ├── util.py # полезные модели
│   └── waveunet.py
├── noiser.py # модуль по генерации пар данных сигнал - зашумленный сигнал
├── util.py
└── wrap.py # обертка pytorch_ligthing
```

Инструкция по запуску:
```
git clone https://github.com/sir-timio/web-denoiser.git
cd web-denoiser
```

### create conda env
```
conda env create -f environment.yml
conda activate env
```

### generate train data
maybe usefull https://github.com/snakers4/open_stt/#links

- download/get clean and noise audio files. Used data: https://github.com/microsoft/MS-SNSD:
    ```
    git clone https://github.com/microsoft/MS-SNSD
    cp -r MS-SNSD/noise_train wave_wizard/noise_train
    cp -r MS-SNSD/noise_train wave_wizard/clean_train
    ```
- generate train data, split it into train, val and test as you wish, but test metrics take a lot of time
    ```
    cd wave_wizard
    python noiser.py --config configs/noiser_config.yaml
    ```
- run training, check metrics and debug samples
    ```
    python train.py --config configs/train_config.yaml
    ```
- create app/.env file with variables
    ```
    STORAGE_FOLDER=<YOUR STORAGE FOLDER>
    DENOISER_V=dns64
    WHISPER_V=base # from pip install git+https://github.com/openai/whisper.git 
    ```
- lauch app
    ```
    streamlit run app/app.py
    ```

![image](front.jpg)