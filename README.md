# ai-fer-server
A python (Django) server that accepts an image of a face as input and outputs an emotion report

## Running Locally with GC Credentials
In order to run the container locally with the model in GCS, after building the image, run:
``` bash
ADC_HOST=local/path/to/gc_credentials.json \
&& \
ADC_CONTAINER=container/path/to/credentials.json \
&& \
docker run -e GOOGLE_APPLICATION_CREDENTIALS=${ADC_CONTAINER} -v ${ADC_HOST}:${ADC_CONTAINER}:ro ai-fers
```
where `ADC_HOST` is the path to Google Cloud's Application Default Credentials JSON file on the host machine. Above, we are mapping our local (host machine's) ADC to one on the container. `ADC_CONTAINER` can be any arbitrary path.
