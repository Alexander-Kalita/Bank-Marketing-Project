# Use an official Python runtime as a parent image
FROM python:3.8-slim AS base

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Stage for Flask predictions service
FROM base AS flask-service
RUN pip install --no-cache-dir -r requirements.txt numpy==1.19.5 joblib==1.3.2 xgboost==2.0.0 flask==1.1.4 jinja2==2.11.2 markupsafe==1.1.1 scikit-learn==0.24.2
EXPOSE 8081
CMD ["python", "app.py"]

# Stage for Streamlit app
FROM base AS streamlit-app
RUN pip install --no-cache-dir -r requirements.txt pandas==1.3.3 joblib==1.3.2 xgboost==2.0.0 protobuf==3.20.0 altair==4.1.0 vega-datasets==0.9.0 streamlit==1.27.2 requests==2.31.0 gunicorn==20.1.0 gevent==23.7.0 scikit-learn==0.24.2
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]