# Credit Card Defaulter

## Overview
This is a classification model for a most common dataset, Credit Card defaulter prediction. Prediction of the next month credit card defaulter based on demographic and last six months behavioral data of customers.

## Motivation
There are times when even a seemingly manageable debt, such as credit cards, goes out of control. Loss of job, medical crisis or business failure are some of the reasons that can impact your finances. In fact, credit card debts are usually the first to get out of hand in such situations due to hefty finance charges (compounded on daily balances) and other penalties.

A lot of us would be able to relate to this scenario. We may have missed credit card payments once or twice because of forgotten due dates or cash flow issues. But what happens when this continues for months? How to predict if a customer will be defaulter in next months?

To reduce the risk of Banks, this model has been developed to predict customer defaulter based on demographic data like gender, age, marital status and behavioral data like last payments, past transactions etc.

## Dataset Information
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients.

## Technical Aspect
This project is divided into two part:
1. Training a classification model to predict defaulter as accurate as possible.
	- Cleaning the datasets, fixing all features
	- Apply Classification ML model
2. Building and hosting a Flask web app.
	- Build the web app using Flask API
	- Upload the project on GitHub
    - Get the customer information from Web app
    - Display the prediction 

## Installation
The Code is written in Python 3.8. 

```bash
cconda create -p venv python==3.8 -y
```
To create virtual env.

```bash
pip install -r requirements.txt
```
## Directory Tree 
```
├── templates 
│   └── index.html
    └── home.html
├── app.py
├── notebook\data\credit-card-default.csv
├── src
├── artifacts\model.pkl
├── README.md
├── Docs
├── log file
├── setup.py
├── README.md
└── requirements.txt
```
## Credits
- The datasets has been provided by [Kaggle](https://www.kaggle.com/datasets/gauravtopre/credit-card-defaulter-prediction).
 
