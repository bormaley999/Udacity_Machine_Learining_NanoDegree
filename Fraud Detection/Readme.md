# Detecting Payment Card Fraud

This repository contains code and associated files for deploying ML models using AWS SageMaker. This repository consists of a number of tutorial notebooks for various case studies, code exercises, and project files that will be to illustrate parts of the ML workflow and give you practice deploying a variety of ML algorithms.

## Project Description

Detaild project descroption could be found in the Project Notebook in the current folder.
In this section, we'll look at a credit card fraud detection dataset, and build a binary classification model that can identify transactions as either fraudulent or valid, based on provided, *historical* data. In a [2016 study](https://nilsonreport.com/upload/content_promo/The_Nilson_Report_10-17-2016.pdf), it was estimated that credit card fraud was responsible for over 20 billion dollars in loss, worldwide. Accurately detecting cases of fraud is an ongoing area of research.

<img src=notebook_ims/fraud_detection.png width=50% />


**The original source for the project**
[Payment Fraud Detection](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Payment_Fraud_Detection): Learn how to build and deploy a supervised, LinearLearner model in SageMaker. You'll tune a model and handle a case of class imbalance to train a model to detect cases of credit card fraud.

## Labeled Data

The payment fraud data set (Dal Pozzolo et al. 2015) was downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/data). This has features and labels for thousands of credit card transactions, each of which is labeled as fraudulent or valid. In this notebook, we'd like to train a model based on the features of these transactions so that we can predict risky or fraudulent transactions in the future.


## Setup Instructions

The notebooks provided in this repository are intended to be executed using Amazon's SageMaker platform. 