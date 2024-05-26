FROM python:3.10-buster

ADD hfer ./hfer

COPY requirements.txt .
COPY setup.py .

RUN pip install --upgrade pip
RUN pip install .

# This section only applies when running the container in standalone mode.
# When docker-compose is used, the `environment` section of docker-compose.yml
# takes precendence over the `ENV` commands below.
WORKDIR /hfer
RUN mkdir models
COPY config/config.json ./config/
ENV MODEL_PATH="../models/model.h5"
ENV CONFIG_PATH="../config/config.json"
# Comment out the line below if not using GCS
#ENV BUCKET_NAME="fers-bucket"

EXPOSE 8000
WORKDIR /hfer/hfer
CMD uvicorn hfer.server.main_server:app --host 0.0.0.0
