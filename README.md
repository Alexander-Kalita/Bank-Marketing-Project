# Predicting telemarketing success in the banking sector: the case of Portuguese bank

## Project Description
This project focuses on predicting bank deposit subscriptions using marketing campaign data from a Portuguese bank. The work is divided into two comprehensive parts. 

The first part of the project, outlined in ![Kalita-A-Bank-marketing-ML](https://github.com/Alexander-Kalita/Bank-Marketing-Project/blob/main/Kalita-A-Bank-marketing-ML.ipynb), is dedicated to preparing the Machine Learning (ML) model. This includes describing the dataset, exploring the data, cleaning it, and conducting preliminary analysis. It proceeds with training and evaluating several classification models, ultimately selecting the XGBoost model for its balanced performance, achieving 88.4% accuracy, 85% ROC AUC, and 80% TPR at 10% FPR.

The second part of the project, presented in ![Kalita-A-Bank-marketing-MLOps](https://github.com/Alexander-Kalita/Bank-Marketing-Project/blob/main/Kalita-A-Bank-marketing-MLOps.ipynb), focuses on developing and deploying an ML application named Depositer, based on the best-trained XGBoost model. This phase involves implementing version control, managing dependencies, and conducting local testing to ensure robustness and functionality. In the staging environment, the model is prepared for production by containerizing it and verifying its performance, ensuring the application runs smoothly in a production-like setting.

The Depositer application addresses the challenge of efficiently targeting customers who are likely to subscribe to term deposits, thereby optimizing marketing campaigns. It has been rigorously tested and functions as expected in both development and staging environments.

## Benefits
The Depositer application is designed to assist telemarketers in targeting potential subscribers during deposit marketing campaigns, minimizing inefficiencies in identifying the desired customer group.

Depositer leverages various customer features such as age, occupation, and call duration to differentiate potential bank deposit subscribers from non-subscribers. By utilizing this predictive capability, users can optimize their marketing efforts, targeting individuals more likely to subscribe, thus saving time and marketing resources.

Depositer is built for speed, simplicity, and ease of use, featuring a user-friendly, minimalist UI. Below is the demo video showcasing the interactive Depositer application in the pre-production stage.


![Depositer Demo](https://github.com/Alexander-Kalita/Bank-Marketing-Project/blob/main/depositer_staging.gif) 

## Data
The dataset utilized in this study was gathered by a Portuguese retail bank from May 2008 to November 2010 and is publicly accessible via the UCI Machine Learning Repository. The data can be accessed ![here](https://github.com/Alexander-Kalita/Bank-Marketing-Project/tree/main/my_mlops_project/data). It comprises 45,211 instances, detailing socio-demographic characteristics, financial backgrounds, and interactions from telemarketing campaigns.

Two notable limitations of this dataset should be considered. First, its collection dates back over a decade, which may impact its applicability to contemporary banking practices. Second, being static, it does not reflect changes in customer behavior or market trends over time. Despite these constraints, the dataset remains a valuable resource for analyzing customer behavior and evaluating the effectiveness of telemarketing campaigns.

## Architecture
The project architecture is outlined below. It begins with uploading the dataset and conducting exploratory data analysis in JupyterLab, where the XGBoost model is identified as the best-performing model. This phase marks the first part of the project.

Following this, the trained XGBoost model is managed with version control using Git, and various experiments are carried out with MLflow within a Pipenv virtual environment. The model undergoes local testing using a Flask server, where essential feature reliability tests are performed, including explanations with SHAP values and bias assessments. Additionally, the model is demonstrated in a local environment via a Streamlit UI to showcase the Depositer application.

In the pre-production stage, the focus shifts to preparing the model for a production-ready environment. This involves deploying the model with Gunicorn and Streamlit, followed by comprehensive integration testing. Both the server and client are containerized using Docker, uploaded to Docker Hub, and connected using Docker Compose.


![Depositer architecture](https://github.com/Alexander-Kalita/Bank-Marketing-Project/blob/main/Depositer_dev_staging.svg).



