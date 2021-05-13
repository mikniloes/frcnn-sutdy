FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
 
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install vim wget git curl sudo python3.8-dev libgl1-mesa-glx python3.8-distutils libxml2-dev libxslt1-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN sudo python3.8 get-pip.py

RUN pip install colorama easydict pyyaml matplotlib numpy opencv-python torch==1.4.0 torchvision==0.5.0
RUN cd /root
RUN git clone https://github.com/mikniloes/pytorch-frcnn-test.git
RUN cd /root/pytorch-frcnn-test
RUN mkdir data
RUN mkdir data/pretrained_model
RUN wget -O data/pretrained_model/resnet50.pth

RUN cd lib
RUN python3.8 setup.py develop



