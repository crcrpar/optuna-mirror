FROM python:3.5-slim-buster

WORKDIR /usr/src/

RUN pip install --no-cache-dir optuna psycopg2-binary tensorflow mlflow

COPY tf_keras_distributed.py .

CMD ["python", "tf_keras_distributed.py"]
