FROM python:3.13-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
