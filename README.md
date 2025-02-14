# 📈 Stock Market Analysis Using Heuristic Approach: LSTM-Based Predictive Modeling

## 🔍 Overview
Stock market prediction remains a challenging task due to its highly non-linear, volatile, and complex nature. Traditional models such as **Moving Averages (MA)** and **Autoregressive Integrated Moving Average (ARIMA)** often fail to capture long-term dependencies in stock price movements. 

This project leverages **Long Short-Term Memory (LSTM)** networks, a specialized type of **Recurrent Neural Network (RNN)**, to model sequential dependencies in stock prices effectively. 

### 📌 Key Contributions
- **LSTM-Based Predictive Model:** Captures long-term dependencies in stock price sequences.
- **Comprehensive Data Pipeline:** Ensures clean and well-processed historical market data.
- **Robust Model Evaluation:** Compares LSTM with traditional methods (MA, ARIMA, and Prophet).
- **Real-World Case Studies:** Tested on market events like **Black Monday (1987), COVID-19 crash (2020), and Fed rate hikes (2022).**

---
## 🏆 Key Features
- **Advanced Temporal Modeling:** Leverages LSTM’s gated memory units to learn from past stock prices.
- **Data Preprocessing & Feature Selection:**
  - Data sourced from **Yahoo Finance API, NASDAQ-100, and additional sentiment datasets.**
  - Key features include **OHLC (Open, High, Low, Close), Volume, Moving Averages (50-day, 200-day), and RSI (Relative Strength Index).**
  - Normalization via **Min-Max scaling** for stable training.
- **Model Interpretability:** Uses **SHAP (SHapley Additive exPlanations)** to understand feature importance.
- **Graphical Evaluation:** Visualization of **actual vs predicted stock prices** using performance plots.

---
## 📊 Methodology
### 🔹 Data Collection & Preprocessing
| Step | Description | Tools Used |
|------|------------|------------|
| **Data Collection** | Gathered OHLCV stock price data (2000–2020) from Yahoo Finance. | `yfinance`, `pandas` |
| **Cleaning** | Removed missing values and 0.8% outliers using the **Interquartile Range (IQR) method**. | `scipy`, `numpy` |
| **Normalization** | Scaled features to **[0,1]** range using **MinMaxScaler**. | `sklearn.preprocessing` |
| **Sequencing** | Created input-output pairs with a **60-day lookback window**. | `tf.keras.TimeseriesGenerator` |

### 🔹 Model Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(60, 6)),  
    Dropout(0.3),  
    LSTM(64, return_sequences=False),  
    Dropout(0.2),  
    Dense(32, activation='relu'),  
    Dense(1)  
])
```

### 🔹 Hyperparameters
- **Optimizer:** Adam (`learning_rate=0.001`)
- **Loss Function:** Huber Loss (`delta=1.5`)
- **Batch Size:** 64
- **Epochs:** 200 (Early Stopping with `patience=15`)
- **Training Hardware:** **NVIDIA RTX 3090 GPU**

---
## 📈 Results
### Model Performance Comparison
| Model | MAE | RMSE | R² Score | Training Time (min) |
|-------|------|------|----------|---------------------|
| **LSTM (Proposed Model)** | **1.17** | **1.82** | **0.991** | **45** |
| ARIMA | 1.49 | 2.31 | 0.975 | 2 |
| Prophet | 2.01 | 3.12 | 0.942 | 8 |
| Moving Average | 2.49 | 3.45 | 0.927 | <1 |

### 📈 Visualization of Predicted vs. Actual Stock Prices
<p align="center">
    <img src="https://github.com/user-attachments/assets/7e5e992f-75e4-41ef-acf6-4d0c513714af" width="600" />
</p>
<p align="center"><strong>Figure 1:</strong> LSTM Model Predictions vs Actual Closing Prices</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/38094588-eaf0-4b88-a341-7ac07cc475c8" width="600" />
</p>
<p align="center"><strong>Figure 2:</strong> Predicted vs Actual Stock Prices (Blue: Actual, Red: Predicted)</p>

---
## 🚀 Future Enhancements
- **Sentiment Analysis Integration:** Incorporate **BERT embeddings** to analyze financial news headlines for better stock predictions.
- **Hybrid Models:** Combine **CNNs (for candlestick pattern recognition) with LSTMs.**
- **Real-Time Deployment:** Implement a **Flask/Dash-based dashboard** for live stock price forecasts.
- **Quantum AI Integration:** Experiment with **Quantum LSTMs for financial market modeling.**

---
## 👥 Authors & Acknowledgments
### Researchers:
- **Shreyas Khandale** ([GitHub](https://github.com/yourusername))
- **Prathamesh Patil** ([GitHub](https://github.com/PrathameshPatil547))
- **Rohan Patil** ([GitHub](https://github.com/rohanpatil2))

**Published in**: IJRASET Volume 11, Issue X, October 2023

### 📜 Citation (BibTeX)
```bibtex
@article{khandale2023stock,  
  title={Stock Market Analysis Using Heuristic Approach: LSTM Networks for Predictive Modeling},  
  author={Khandale, Shreyas and Patil, Prathamesh and Patil, Rohan},  
  journal={IJRASET},  
  volume={11},  
  number={X},  
  pages={112--120},  
  year={2023}  
}  
```

---
## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---
## 💡 Getting Started
### Installation & Usage
```bash
git clone https://github.com/yourusername/Stock-Market-Analysis-using-Heuristic-Approach.git  
cd Stock-Market-Analysis-using-Heuristic-Approach  
pip install -r requirements.txt  # TensorFlow, yfinance, pandas, numpy  
python train.py --epochs 200 --batch_size 64  
```

### Run Predictions
```bash
python predict.py --model saved_model.h5 --data test_data.csv
```

---
## 🤝 Contributing
We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit changes and push to your branch.
4. Open a pull request (PR).

## 📫 Contact
📩 Email: [your-email@example.com](mailto:your-email@example.com)
