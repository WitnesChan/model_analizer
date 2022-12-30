# syntax=docker/dockerfile:1

FROM python:3.9.16-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD [ "python3" , "base/tune.py"]

