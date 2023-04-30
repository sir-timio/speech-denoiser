docker run -p 8501:8501 \
    --name denoiser \
    -v ~/.config/pulse:/home/pulseaudio/.config/pulse \
    -it denoiser