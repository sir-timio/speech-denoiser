for both web and live, specify yaml config in configs/

On mac os, install [BlackHole](https://github.com/ExistentialAudio/BlackHole) first.

# live
```
    python live/run.py --config configs/live.yaml
```

Then select chosen channel as mic.

![image](../doc/img/mic_choice.jpg)

# web
```
    streamlit run web/app.py -- --config configs/web.yaml
```
![image](../doc/img/front.jpg)
