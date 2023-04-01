FROM python:3.10-buster

WORKDIR /usr/src/app

RUN pip install --upgrade pip
COPY reqq.txt .
RUN pip install -r reqq.txt

# Install sugar metaplex

COPY . .
