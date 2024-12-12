FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y libpq-dev python-dev-is-python3

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/launch_server.py"]