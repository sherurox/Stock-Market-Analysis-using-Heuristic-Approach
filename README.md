# Stock Market Analysis Using Heuristic Approach: LSTM-Based Predictive Modeling  
![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.1-orange) 
[![Paper](https://img.shields.io/badge/Published%20in-IJRASET%202023-brightgreen)](https://www.ijraset.com/best-journal/stock-market-analysis-using-heuristic-approach)

**Accurate Stock Price Forecasting with Deep Learning**  
*Leveraging 20+ Years of S&P 500 Data for High-Precision Market Trend Prediction*

---

## 📌 Table of Contents
- [Abstract](#-abstract)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Enhancements](#-future-enhancements)
- [Authors](#-authors)
- [License](#-license)

---

## 🔍 Abstract
This project implements an **LSTM (Long Short-Term Memory)** neural network to predict stock market trends using historical S&P 500 data (2000–2020). Outperforming traditional models like ARIMA and Moving Average, our solution achieves:

| Metric                | LSTM   | ARIMA  | Moving Average |
|-----------------------|--------|--------|----------------|
| **MAE**               | 1.17   | 1.49   | 2.49           |
| **Correlation (r)**   | 0.996  | 0.983  | 0.944          |

![Prediction Demo](https://github.com/user-attachments/assets/38094588-eaf0-4b88-a341-7ac07cc475c8)  
*Figure: Actual vs Predicted Closing Prices (Test Set)*

---

## 🏆 Key Features
- **Advanced Temporal Modeling**: LSTM architecture with 128-64 neuron layers.
- **Comprehensive Data Pipeline**:
  - **Sources**: Yahoo Finance API, NASDAQ-100.
  - **Features**: OHLC prices, Volume, RSI, Moving Averages.
- **Robust Preprocessing**: Outlier removal, Min-Max normalization, 60-day lookback windows.
- **Model Interpretability**: SHAP values for feature importance analysis.

---

## 📊 Methodology

### Data Preprocessing Pipeline
| Step               | Tools/Techniques                              | Output Shape      |
|--------------------|-----------------------------------------------|-------------------|
| Data Collection    | `yfinance` API                                | (5043, 6)         |
| Outlier Removal    | IQR (Interquartile Range)                     | 98.2% data retained |
| Normalization      | MinMaxScaler (0–1 range)                      | (5043, 6)         |
| Sequencing         | 60-day lookback window                        | (4983, 60, 6)     |

### Model Architecture
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(60, 6)),  
    Dropout(0.3),  
    LSTM(64, return_sequences=False),  
    Dropout(0.2),  
    Dense(32, activation='relu'),  
    Dense(1)  
])
Hyperparameters:

Optimizer: Adam (lr=0.001)

Loss: Huber Loss (delta=1.5)

Batch Size: 64

Training Time: 45 mins (NVIDIA RTX 3090)

📈 Results
Performance Comparison
Model	MAE	RMSE	R² Score
LSTM (Proposed)	1.17	1.82	0.991
ARIMA	1.49	2.31	0.975
Prophet	2.01	3.12	0.942
Feature Importance (SHAP Values)
Feature	Impact Score
Previous Close	0.41
50-day MA	0.29
RSI	0.18
Volume	0.12
💻 Installation
Clone the repository:

bash
Copy
git clone https://github.com/yourusername/Stock-Market-Analysis-using-Heuristic-Approach.git
cd Stock-Market-Analysis-using-Heuristic-Approach
Install dependencies:

bash
Copy
pip install -r requirements.txt  # TensorFlow, yfinance, pandas, numpy
🚀 Usage
Train the model:

bash
Copy
python train.py --epochs 200 --batch_size 64
Predict on new data:

python
Copy
from predict import StockPredictor
predictor = StockPredictor("models/lstm_model.h5")
predictor.predict_next_day("AAPL")
🔮 Future Enhancements
Integrate news sentiment analysis using BERT.

Deploy real-time dashboard with Flask/Dash.

Experiment with Quantum Machine Learning.

👥 Authors
Shreyas Khandale - GitHub

Prathamesh Patil - GitHub

Rohan Patil - GitHub

Published in: IJRASET, Oct 2023
Cite this work:

bibtex
Copy
@article{khandale2023stock,
  title={Stock Market Analysis Using Heuristic Approach: LSTM Networks for Predictive Modeling},
  author={Khandale, Shreyas and Patil, Prathamesh and Patil, Rohan},
  journal={IJRASET},
  volume={11},
  pages={112--120},
  year={2023}
}
📜 License
This project is licensed under the MIT License. See LICENSE for details.

Copy

---

**🌟 Star this repo if you find it useful!**  
**🐛 Report issues [here](https://github.com/yourusername/Stock-Market-Analysis-using-Heuristic-Approach/issues).**
