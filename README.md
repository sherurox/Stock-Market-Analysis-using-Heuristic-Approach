# Stock-Market-Analysis-using-Heuristic-Approach
This repository contains the code and documentation for Stock Market Analysis using Heuristic Approach, which explores the use of Long Short-Term Memory (LSTM) networks for stock price prediction. The research is based on historical stock data, and the model aims to provide more accurate market trend analysis compared to traditional models. <br>

## 🔍 Abstract
The stock market is a crucial financial sector that influences economic activities. The project implements LSTM-based predictive modeling, leveraging Recurrent Neural Networks (RNNs) to analyze and forecast stock price trends. The proposed approach aims to outperform traditional market analysis models such as Moving Average and ARIMA by capturing complex patterns in stock price fluctuations.<br>

## 🏆 Key Features
Utilizes LSTM Neural Networks for stock price prediction.<br>
Uses historical stock price data for training and validation.<br>
Compares performance with Moving Average and ARIMA models.<br>
Implements data preprocessing, feature selection, and model evaluation.<br>
Provides real-world case studies for validation.<br>

## 📊 Methodology
The project follows a systematic approach in data collection, preprocessing, feature selection, and model training:<br>

Data Collection<br>
<br>
Sources: Historical stock price datasets (e.g., S&P 500 Index).<br>
Timeframe: January 2000 - December 2020.<br>
Features: Stock price, trading volume, market trends.<br>
Data Preprocessing<br>
<br>
Cleaning missing values, duplicates, and outliers.<br>
Standardizing and normalizing data for better performance.<br>
Splitting dataset: 80% Training, 20% Testing.<br>
Model Development<br>
<br>
LSTM-based Recurrent Neural Network (RNN) architecture.<br>
Training with Backpropagation & Adam Optimizer.<br>
Hyperparameter tuning: Batch size, Learning rate, Epochs.<br>
Model Evaluation<br>
<br>
Performance Metrics: MAE, RMSE, Correlation Coefficient.<br>
Baseline comparison: Moving Average, ARIMA.<br>
Graphical representation of actual vs predicted values.<br>
<br>

## 📈 Results

The performance of the **LSTM model** was evaluated against traditional forecasting models, including **Moving Average** and **ARIMA**. The results demonstrate that LSTM outperforms these methods in terms of **Mean Absolute Error (MAE)** and **Correlation Coefficient**.

### 📊 Model Performance Comparison

| Model          | Mean Absolute Error (MAE) | Correlation Coefficient (r) |
|---------------|--------------------------|------------------------------|
| **LSTM (Proposed Model)** | **1.17** | **0.996** |
| Moving Average | 2.49 | 0.944 |
| ARIMA | 1.49 | 0.983 |

### 📈 Visualization of Predicted vs. Actual Stock Prices

<p align="center">
    <img src="https://github.com/user-attachments/assets/7e5e992f-75e4-41ef-acf6-4d0c513714af" width="400" />
</p>
<p align="center"><strong>Figure 1:</strong> The above diagram depicts the predicted values.</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/38094588-eaf0-4b88-a341-7ac07cc475c8" width="400" />
</p>
<p align="center"><strong>Figure 2:</strong> The above graph depicts the relation between predicted and actual values.<br>(Blue-Actual price, Red-Predicted price)</p>

### 📌 Key Observations
- The **LSTM model** achieved the **lowest MAE (1.17)**, indicating a more accurate prediction.
- LSTM’s **correlation coefficient (0.996)** shows a near-perfect relationship with actual stock prices.
- The **Moving Average and ARIMA models** performed reasonably well but showed higher errors compared to LSTM.


## 📌 Key Observations
- The **LSTM model** achieved the **lowest MAE (1.17)**, indicating a more accurate prediction.<br>
- LSTM’s **correlation coefficient (0.996)** shows a near-perfect relationship with actual stock prices.<br>
- The **Moving Average and ARIMA models** performed reasonably well but showed higher errors compared to LSTM.<br>

# Future Enhancements
Integrate Sentiment Analysis for improved stock price forecasting.<br>
Implement Hybrid AI Models (CNN + LSTM).<br>
Develop a real-time stock market prediction dashboard.<br>
Optimize model for low-latency, high-frequency trading.<br>

## Authors & Acknowledgments
This research was conducted by Shreyas Khandale (me), <br>
Prathamesh Patil (https://github.com/PrathameshPatil547), and  <br>
Rohan Patil (https://github.com/rohanpatil2), published in IJRASET, October 2023. <br>

🔗 Paper Link: Predicting Credit Card Defaults with Machine Learning <br>
https://www.ijraset.com/best-journal/stock-market-analysis-using-heuristic-approach

# License
This project is licensed under the MIT License - see the LICENSE file for details.

















