FROM python:3.10.14-slim-bullseye

WORKDIR /app

COPY requirements.txt .

ENV TMPDIR=/data/vincents/

RUN apt-get update && \
    apt-get -y upgrade && \
    pip install --cache-dir=/data/vincents/ -b /data/vincents/ -r requirements.txt

EXPOSE 8501

COPY . /app