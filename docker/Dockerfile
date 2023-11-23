FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -y
RUN apt install -y sudo git make curl ssh vim python3 python3-pip libgl1-mesa-dev python3-tk
RUN pip install opencv-python opencv-contrib-python numpy matplotlib
RUN useradd -m -s /bin/bash hoge
RUN echo "hoge:hogehoge" | chpasswd
RUN gpasswd -a hoge sudo

USER hoge

workdir /osero
