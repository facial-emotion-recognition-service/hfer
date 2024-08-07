FROM python:3.10-buster
RUN apt-get -y update && apt-get install -y build-essential cmake libhdf5-dev

ADD hfer ./hfer

COPY requirements.txt .
COPY setup.py .
COPY models .
COPY README.md .


RUN pip install --upgrade pip
RUN pip install --no-binary h5py h5py
RUN pip install .

# This section only applies when running the container in standalone mode.
# When docker-compose is used, the `environment` section of docker-compose.yml
# takes precendence over the `ENV` commands below.

ENV MODEL_PATH="../../model.h5"
# Comment out the line below if not using GCS
#ENV BUCKET_NAME="fers-bucket"

# EXPOSE 8000
WORKDIR /hfer/hfer
CMD uvicorn hfer.server.main_server:app --host 0.0.0.0 --port $PORT
