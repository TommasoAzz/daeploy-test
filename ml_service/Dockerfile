FROM python:3.9.7

RUN mkdir /ml_service/
WORKDIR /ml_service/

COPY requirements.txt /ml_service/

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /ml_service/

EXPOSE 3000

ENTRYPOINT ["python3"]

CMD ["train_classifier.py"]