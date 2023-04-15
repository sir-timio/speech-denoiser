# Web speech denoiser and transcriber

Краткий отчет о проведенных эксприментах:

Было реализованно 2 варианта архитектура: Wave-Unet и Demucs
архитектура взята из https://github.com/facebookresearch/denoiser
Wave-Unet адаптированно со статьи https://arxiv.org/pdf/1806.03185.pdf

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