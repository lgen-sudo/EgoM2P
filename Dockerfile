FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=http://proxy.cscs.ch:8080
ENV https_proxy=https://proxy.cscs.ch:8080

RUN apt-get update && apt-get install -y python3-venv ffmpeg libsm6 libxext6 libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev && apt-get clean && rm -rf /var/lib/apt/lists/*