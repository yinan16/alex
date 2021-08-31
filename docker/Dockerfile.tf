FROM tensorflow/tensorflow:latest-gpu

RUN apt update && apt install -y graphviz python3-venv

RUN python3 -m pip install --upgrade pip build
