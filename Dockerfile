FROM python:3
USER root
WORKDIR /root/src/

# タイムゾーン
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# apt
RUN apt update
# RUN apt install -y libopencv-dev
RUN apt install -y libopencv-dev opencv-data

ADD requirements.txt /root/src/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt