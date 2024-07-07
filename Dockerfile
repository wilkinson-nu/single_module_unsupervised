FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update
RUN apt-get install -y emacs
RUN python -m pip install --upgrade pip
RUN pip install h5py
RUN pip install pyqt5
RUN apt-get install -y libgl1 
RUN apt-get install -y libxcb-xinerama0 ## NECESSARY!
RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
RUN apt-get install -y ffmpeg

ENV XDG_RUNTIME_DIR=${PWD}
ENV export NO_AT_BRIDGE=1