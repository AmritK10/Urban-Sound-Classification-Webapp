FROM python:3.6.10
WORKDIR /usr/src/app
ENV FLASK_APP flask_app.py
ENV FLASK_RUN_HOST 0.0.0.0
COPY requirements.txt requirements.txt
RUN apt update
RUN apt-get install libsndfile1 -y
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir static
COPY . .
EXPOSE 5000
CMD ["flask", "run"]