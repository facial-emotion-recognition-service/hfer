FROM python:3.10-buster

ADD ai_fers ./ai_fers

COPY requirements.txt .
COPY setup.py .

RUN pip install --upgrade pip
RUN pip install .

# This section only applies when running the container in standalone mode.
# When docker-compose is used, the `environment` section of docker-compose.yml
# takes precendence over the `ENV` commands below.
WORKDIR /ai_fers
RUN mkdir input_images
RUN mkdir output_json
RUN mkdir models
COPY config/config.json ./config/
ENV IMAGE_INPUT_DIR="../input_images/"
ENV JSON_OUTPUT_DIR="../output_json/"
ENV MODEL_PATH="../models/model.h5"
ENV CONFIG_PATH="../config/config.json"
# Comment out the line below if not using GCS
ENV BUCKET_NAME="fers-bucket"

EXPOSE 8000
WORKDIR /ai_fers/server
CMD ["python", "main_server.py", "runserver", "0.0.0.0:8000"]
