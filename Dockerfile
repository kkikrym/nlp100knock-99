FROM kkikrym/nlp100-99:latest

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get -y upgrade && \
    pip install -r requirements.txt

EXPOSE 8501

COPY . /app