# Create Dockerfile for Flask api
FROM python:3.8-slim
WORKDIR /app
COPY flask_api /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8081
CMD ["python", "app.py"]