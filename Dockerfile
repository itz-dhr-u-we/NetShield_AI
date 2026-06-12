FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y curl unzip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip install awscli

CMD ["python3", "app.py"]