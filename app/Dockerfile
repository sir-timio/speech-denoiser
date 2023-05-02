# syntax = docker/dockerfile:experimental
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg

# for mic
# RUN apt-get install -y alsa-base alsa-utils libsndfile1-dev && apt-get clean
# for macos
RUN apt-get install -y pulseaudio
ENV PULSE_SERVER=docker.for.mac.localhost


COPY requirements.txt requirements.txt
COPY . .

WORKDIR /

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip3 install -r requirements.txt
# RUN pip3 install -r requirements.txt

RUN pip install --upgrade pip wheel

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "web/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--", "--config", "configs/web.yaml"]