# Predicting telemarketing success in the banking sector: the case of Portuguese bank

## Description
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
The dataset used in this study was collected by a Portuguese retail bank between May 2008 and November 2010, and is publicly available in the ![UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). The data used in this study could be found ![here](https://github.com/Alexander-Kalita/Bank-Marketing-Project/tree/main/my_mlops_project/data). The dataset contains information on 45,211 instances, including socio-demographic characteristics, financial backgrounds, and telemarketing campaign interactions.

There are two main limitations of the dataset that should be noted. First, the data was collected over a decade ago, which may limit its relevance to current banking practices. Second, the dataset is static and does not account for changes in customer behavior or market trends over time. However, despite these limitations, the dataset provides a valuable source of information on customer behavior and telemarketing campaign effectiveness.





