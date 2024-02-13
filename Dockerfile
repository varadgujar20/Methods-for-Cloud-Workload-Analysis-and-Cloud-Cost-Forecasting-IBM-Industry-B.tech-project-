FROM python:3.7.4

RUN apt-get update

ENV APP_HOME /app
WORKDIR ${APP_HOME}

COPY . ./

RUN ls

ENV PYTHONUNBUFFERED True

RUN pip install -r requirements.txt

#Execute app
CMD exec gunicorn --bind 0.0.0.0:8080 --workers 4 --timeout 0 app:app