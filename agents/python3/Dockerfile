FROM python:3.8-bullseye

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN python -m pip install -r requirements.txt

COPY . /app
ENTRYPOINT ["python", "train.py"]
