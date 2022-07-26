# Dockerfile to setup JAVA_11 container with custom user and some dependecies installed
FROM openjdk:11

# two lines above are required for minicoda setup
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

LABEL maintainer="tprakap@coli.uni-saarland.de"

# install required utils
RUN apt-get -y update

RUN apt-get -y install git
RUN apt-get -y install screen
RUN apt-get -y install python2
RUN apt-get -y install vim
RUN apt-get -y install g++
RUN apt-get -y install gcc

# setup miniconda
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda install python=3.7
# RUN conda --version
RUN . /root/miniconda3/bin/activate
# RUN conda list python -f
RUN conda install -y Cython
RUN conda install -y -c conda-forge jsonnet

# allennlp recommends installing pytorch first if you're installing allennlp via pip:
# https://pypi.org/project/allennlp/#installing-via-pip
# instructions for conda install old versions of pytorch (am-parser documentation recommends 1.1)
# https://pytorch.org/get-started/previous-versions/
RUN conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# install am-parser requirements
COPY requirements.txt /am-parser-app/requirements.txt
WORKDIR /am-parser-app

RUN pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_md
COPY . /am-parser-app

CMD [ "bash" ]
