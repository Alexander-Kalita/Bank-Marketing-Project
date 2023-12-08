FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt numpy==1.19.5 joblib==1.3.2 xgboost==2.0.0 flask==1.1.4 jinja2==2.11.2 markupsafe==1.1.1 scikit-learn==0.24.2
EXPOSE 8081
CMD ["python", "app.py"]