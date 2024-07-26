# Create Dockerfile for gunicorn_app
FROM python:3.8-slim
WORKDIR /app
COPY gunicorn_app.py /app
COPY xgb_cv.json /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8081
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8081", "gunicorn_app:application"]