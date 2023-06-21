# FROM rasa/rasa:latest-full
FROM python:3.7.6

USER root

COPY . /app/

WORKDIR /app/nlu-engine

EXPOSE 5005

RUN pip install -r requirements.txt

RUN rasa train
ENTRYPOINT ["rasa", "run", "--enable-api", "--debug"]
