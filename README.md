# Predicting Laptop Price Tiers in R Using Machine Learning  
### Author: Salvatore Zizzi  

![R Version](https://img.shields.io/badge/R-%3E%3D%204.0-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow)
[![Author](https://img.shields.io/badge/Author-%20Salvatore_Zizzi-1DA1F2.svg)](https://www.linkedin.com/in/salvatore-zizzi-242151107/)

## Project Description
This project aims to classify laptops into two categories — **High-end** and **Low-end** — based on their technical specifications and price, using machine learning techniques in R.

The dataset includes specs of laptops sold in India, with prices in Indian Rupees. The classification uses the average price as a threshold to determine the **High-end** and **Low-end** categories.

## Dataset
## Dataset Source
The dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/datasets/eslamelsolya/laptop-price-prediction).  
Please download it from there and place the file in the `Data/` folder before running the scripts.


The dataset includes variables such as:
- **Brand, model, and type**  
- **CPU, GPU, and memory specifications**  
- **Operating system**  
- **Price** (used as a qualitative target variable)  

## Preprocessing
- **Missing Data Imputation**: Imputed missing values for the `opsys` (Operating System) and `memory` columns using the median value for each respective category.
- **Categorical Variables**: Optimized the grouping of categorical variables such as CPU and GPU models for better model performance.
- **Feature Removal**: Removed highly correlated and near-zero variance features to improve model accuracy and avoid overfitting.

## Models Used
- **Feature Selection**: Used a Decision Tree for feature selection to identify the most relevant features.
- **Trained Models**:
  - Random Forest  
  - Generalized Linear Models (GLM), Partial Least Squares (PLS), Lasso  
  - K-Nearest Neighbors (KNN), AdaBoost  
  - C5.0, Gradient Boosting  
  - NodeHarvest, PCA Neural Network  
  - Neural Networks with varying hidden layers

## Evaluation
- **Evaluation Metrics**:  
  - ROC Curve  
  - Lift Curve  
  - Confusion Matrix  
- **Selected Model**: **C5.0**  
  The **C5.0** model showed the best trade-off between sensitivity and specificity, offering a balanced performance across multiple metrics. It was selected as the final model.

## Results
- The **C5.0** model achieved the most stable and balanced performance, with a high accuracy rate, precision, and recall.
- Neural networks demonstrated competitive ROC scores but had lower sensitivity, making them less suitable compared to the **C5.0** model for this specific task.

## Requirements
- **R** (version ≥ 4.0)
- Key Packages:
  - `caret`
  - `rpart`
  - `C50`
  - `randomForest`
  - `gbm`
  - `nnet`
  - `pROC`
  - `ggplot2`

## How to Run the Project
1. Place the dataset in the `Data/` folder.
2. Run the main script (`Classification.R`, in the `Code/` folder) in RStudio or any R environment.
3. Output results, including model performance metrics and plots, will be saved in the `output/` folder.

