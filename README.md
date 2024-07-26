# Predicting telemarketing success in the banking sector: the case of Portuguese bank

## Summary
This project focuses on predicting bank deposit subscriptions using marketing campaign data from a Portuguese bank. The work is divided into two comprehensive parts. 

The first part of the project, outlined in ![Kalita-A-Bank-marketing-ML](https://github.com/Alexander-Kalita/Bank-Marketing-Project/blob/main/Kalita-A-Bank-marketing-ML.ipynb), is dedicated to preparing the Machine Learning (ML) model. This includes describing the dataset, exploring the data, cleaning it, and conducting preliminary analysis. It proceeds with training and evaluating several classification models, ultimately selecting the XGBoost model for its balanced performance, achieving 88.4% accuracy, 85% ROC AUC, and 80% TPR at 10% FPR.

The second part of the project, presented in ![Kalita-A-Bank-marketing-MLOps](https://github.com/Alexander-Kalita/Bank-Marketing-Project/blob/main/Kalita-A-Bank-marketing-MLOps.ipynb), focuses on developing and deploying an ML application named Depositer, based on the best-trained XGBoost model. This phase involves implementing version control, managing dependencies, and conducting local testing to ensure robustness and functionality. In the staging environment, the model is prepared for production by containerizing it and verifying its performance, ensuring the application runs smoothly in a production-like setting.

The Depositer application addresses the challenge of efficiently targeting customers who are likely to subscribe to term deposits, thereby optimizing marketing campaigns. It has been rigorously tested and functions as expected in both development and staging environments.
