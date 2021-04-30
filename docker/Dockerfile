ARG CUDA="10.1"

FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libyaml-dev vim zsh wget htop tmux

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

# Create a Python 3.7 environment
RUN /root/miniconda3/bin/conda install -y conda-build \
 && /root/miniconda3/bin/conda create -y --name py37 python=3.7 \
 && /root/miniconda3/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/root/miniconda3/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install requests ninja yacs numpy cython matplotlib opencv-python-headless tqdm \
 progressbar protobuf tensorboardx tensorboard tensorflow nltk pymongo pandas joblib \
 sklearn boto3 easydict scikit-image anytree timm einops
RUN pip --no-cache-dir install --force-reinstall -I pyyaml

RUN python -m nltk.downloader punkt

# Install PyTorch 1.7
ARG CUDA
RUN conda install pytorch torchvision cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
ENV MKL_THREADING_LAYER="GNU"

RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN echo """export ZSH="/root/.oh-my-zsh"\nZSH_THEME="gentoo"\nplugins=(git)\nsource \$ZSH/oh-my-zsh.sh""" > /root/.zshrc
RUN echo """syntax on\nfiletype indent on\nset autoindent\nset number\ncolorscheme desert""" > /root/.vimrc

CMD [ "zsh" ]

