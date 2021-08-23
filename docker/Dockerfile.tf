FROM tensorflow/tensorflow:latest-gpu

RUN apt install -y graphviz
RUN python3 -m pip install --upgrade alex-nn==0.1.6 matplotlib
