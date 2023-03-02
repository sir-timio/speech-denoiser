# Web speech denoiser and transcriber

```
git clone https://github.com/sir-timio/web-denoiser.git
cd web-denoiser
```

### create conda env
```
conda env create -f environment.yml
conda activate env
```

### tune/train on https://www.kaggle.com/datasets/tapakah68/audio-dataset
```
kaggle datasets download -d tapakah68/audio-dataset

git clone https://github.com/facebookresearch/denoiser
cd denoiser
```
then follow readme of denoiser 

### run streamlit app
```
streamlit run app.py
```

![image](front.jpg)