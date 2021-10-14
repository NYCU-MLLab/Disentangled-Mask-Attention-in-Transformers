FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
LABEL authors="Andy Huang"

ARG NUM_CPU=4
ARG TH_VERSION="1.9.0"
ARG DEBIAN_FRONTEND=noninteractive

# install cmake, sox, sndfile, ffmpeg, flac, ATLAS, openssh-server
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python2.7 python3.8 python3-pip git-all cmake automake patch grep \
                       bzip2 unzip wget gfortran libtool subversion gawk sox tmux vim \
                       libsndfile1-dev ffmpeg flac libatlas-base-dev ssh openssh-server \
                       zlib1g-dev bc build-essential libboost-all-dev libbz2-dev liblzma-dev \
                       python-setuptools python3-setuptools sudo libfreetype6-dev

####################################################
# Clone Disentangled Transformer (Comment following 2 lines for clean install)
WORKDIR /root
RUN git clone https://github.com/NYCU-MLLab/Disentangled-Mask-Attention-in-Transformers.git
####################################################

# install Kaldi
WORKDIR /root
RUN git clone https://github.com/kaldi-asr/kaldi
ENV KALDI_ROOT /root/kaldi

WORKDIR /root/kaldi/tools
RUN make -j ${NUM_CPU}

RUN ./extras/install_mkl.sh

WORKDIR /root/kaldi/src
RUN ./configure --use-cuda=no
RUN make -j clean depend; make -j ${NUM_CPU}

# install ESPnet
WORKDIR /root/Disentangled-Transformer/ASR/ESPNet/tools

####################################################
# Clean install (Uncomment)
# 
# WORKDIR /root
# RUN git clone https://github.com/espnet/espnet
# RUN cd ./espnet && git checkout v.0.9.5
# WORKDIR /root/espnet/tools
####################################################

RUN ln -s ${KALDI_ROOT} .

RUN python3 -m pip install -U pip wheel
RUN touch activate_python.sh

ARG CUDA_HOME=/usr/local/cuda
ARG PATH=${CUDA_HOME}/bin:${PATH}
ARG LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ARG CFLAGS="-I${CUDA_HOME}/include ${CFLAGS}"
ARG CPATH=${CUDA_HOME}/include:${CPATH}
ARG CUDA_PATH=${CUDA_HOME}

# chage install warp-rnnt from github to pypi
RUN sed -i '27,33d' installers/install_warp-rnnt.sh
RUN sed -i '26a pip3 install warp_rnnt' installers/install_warp-rnnt.sh
RUN sed -i '15,18d' installers/install_warp-rnnt.sh

# Install pytorch1.9.0+cu11
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# remove chainer-ctc installation in ESPNet makefile
RUN sed -i '182d' Makefile

# remove warp-transducer installation in ESPNet makefile
RUN sed -i '170,174d' Makefile

# remove pytorch installation in ESPNet makefile
RUN sed -i '128d' Makefile

# warp-rnnt may cause not install error in check_install.py
RUN make TH_VERSION=${TH_VERSION}; exit 0

# instasll KenLM
RUN ./installers/install_kenlm.sh

# install fairseq

######################################################
# Clean install (Uncomment)
#
# WORKDIR /root
# RUN git clone https://github.com/pytorch/fairseq

# WORKDIR /root/fairseq
# RUN git checkout v0.10.1
######################################################

WORKDIR /root/Disentangled-Transformer/MT/Fairseq

# install without any cuda extension
RUN sed -i 's/if "CUDA_HOME" in os.environ:/if False:/g' setup.py

RUN pip3 install --editable ./

######################################################
# Comment follwoing 1 line for clean install
RUN pip3 install -r requirements.txt
######################################################

# install apex
WORKDIR /root
RUN git clone https://github.com/NVIDIA/apex

WORKDIR /root/apex

# disable cuda version checking
RUN sed -i "s/check_cuda_torch_binary_vs_bare_metal/# check_cuda_torch_binary_vs_bare_metal/g" setup.py
RUN sed -i "s/def # check_cuda_torch_binary_vs_bare_metal/def check_cuda_torch_binary_vs_bare_metal/g" setup.py

RUN pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
                 --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
                 --global-option="--fast_multihead_attn" ./

ENV CUDA_HOME=/usr/local/cuda

# Install SpeechBrain
WORKDIR /root/Disentangled-Transformer/ASR/SpeechBrain
RUN pip3 install -r requirements.txt
RUN pip3 install --editable .

# install other common tools
RUN pip3 install jupyterlab
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install scikit-learn
RUN pip3 install pyarrow

# tensorboard port
EXPOSE 6006

# jupyter notebook port
EXPOSE 8888

# configure ssh server
# followd the steps described in https://docs.docker.com/engine/examples/running_ssh_service/
RUN mkdir /var/run/sshd
RUN echo "root:root" | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]

WORKDIR /root
