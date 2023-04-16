# Web speech denoiser and transcriber


<!-- краткий отчет о проведенных экспериментах
код обучения моделей
обоснование выбора моделей и гиперпараметров
описание кода обучения -->

Краткий отчет о проведенных эксприментах:

Реализация моделей, цикла обучения представлена в директории wave_wizard.
Было реализованно 3 варианта архитектуры: Wave-Unet и [Demucs](https://github.com/facebookresearch/denoiser) взято практически без изменений, Wave-Unet адаптированно со [статьи](https://arxiv.org/pdf/1806.03185.pdf) и  GateWave - собственная разработка, вдохновленная [статьей](https://paperswithcode.com/method/gated-convolution-network). 
```
.
├── blocks
│   ├── basic.py
│   └── gated.py
├── callbacks.py
├── dataset.py
├── inference.py
├── loss.py
├── models
│   ├── demucs.py
│   ├── gatewave.py
│   ├── resample.py
│   └── waveunet.py
├── noiser.py
├── util.py
└── wrap.py
```

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