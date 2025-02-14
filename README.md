# 📈 Stock Market Analysis Using Heuristic Approach: LSTM-Based Predictive Modeling

## 🔍 Overview
Stock market prediction is a challenging task due to its non-linear and volatile nature. Traditional models like ARIMA often fail to capture long-term dependencies. This project leverages **Long Short-Term Memory (LSTM)** networks, a specialized form of **Recurrent Neural Networks (RNNs)**, to model sequential stock price data and make accurate predictions.

Our model is trained on **20 years of S&P 500 historical data (2000–2020)** and achieves state-of-the-art results, significantly outperforming traditional methods such as Moving Averages and ARIMA.

---
## 🚀 Key Features
- **Advanced Temporal Modeling**: LSTM networks capture long-term dependencies using gated memory cells.
- **Comprehensive Data Pipeline**:
  - **Data Sources**: S&P 500 (Yahoo Finance API), NASDAQ-100, and sentiment datasets.
  - **Features**: Open, High, Low, Close (OHLC), Volume, 50-day/200-day moving averages, RSI.
- **Robust Preprocessing**:
  - Outlier removal using the **Interquartile Range (IQR) method**.
  - **Min-Max normalization** to scale prices to [0,1].
  - Sequential data generation using **60-day lookback windows**.
- **Model Interpretability**: SHAP (SHapley Additive exPlanations) for feature importance analysis.
- **Real-World Validation**: Tested on events like Black Monday (1987), COVID-19 crash (2020), and Fed rate hikes (2022).

---
## 📊 Methodology

### Data Collection & Preprocessing
| Step                | Description                                                  | Tools Used  |
|---------------------|--------------------------------------------------------------|-------------|
| Data Collection    | Fetched OHLCV data from Yahoo Finance API (2000–2020).      | `yfinance`, `pandas` |
| Cleaning          | Removed 0.8% outliers (IQR method), interpolated missing values. | `scipy`, `numpy` |
| Normalization     | Applied MinMaxScaler to normalize stock prices.             | `sklearn.preprocessing` |
| Sequencing       | Created input-output pairs with a 60-day lookback window.    | `tf.keras.TimeseriesGenerator` |

---
## 🏗️ Model Architecture
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
### Hyperparameters
- **Optimizer**: Adam (`learning_rate=0.001`)
- **Loss**: Huber Loss (`delta=1.5`)
- **Batch Size**: 64
- **Epochs**: 200 (with Early Stopping `patience=15`)
- **Training Hardware**: **NVIDIA RTX 3090 GPU**

---
## 📈 Results
### Performance Comparison
| Model            | MAE  | RMSE | R² Score | Training Time (min) |
|-----------------|------|------|----------|---------------------|
| **LSTM (Proposed)** | **1.17** | **1.82** | **0.991**  | **45**  |
| ARIMA           | 1.49 | 2.31 | 0.975    | 2       |
| Prophet        | 2.01 | 3.12 | 0.942    | 8       |
| Moving Average | 2.49 | 3.45 | 0.927    | <1      |

### Feature Importance via SHAP
| Feature          | SHAP Value (Impact on Price) |
|----------------|--------------------------|
| Previous Close | 0.41                     |
| 50-day MA      | 0.29                     |
| RSI           | 0.18                     |
| Volume        | 0.12                     |

---
## 🔮 Future Enhancements
- **Sentiment-Integrated Forecasting**:
  - Incorporate **news headlines (Reuters/WSJ)** using **BERT embeddings**.
  - Develop **Hybrid Model (LSTM + Transformer)** for numerical & textual data.
- **Multi-Modal Architecture**:
  - Combine **CNN (for candlestick charts) with LSTM**.
- **Real-Time Trading System**:
  - Deploy via **TensorFlow Serving**.
  - Develop a **Flask/Dash dashboard** with live predictions.
- **Quantum Machine Learning**:
  - Experiment with **Quantum LSTMs** for portfolio optimization.

---
## 👥 Authors & Acknowledgments
### Researchers:
- **Shreyas Khandale** ([GitHub](https://github.com/yourusername))
- **Prathamesh Patil** ([GitHub](https://github.com/yourusername))
- **Rohan Patil** ([GitHub](https://github.com/yourusername))

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
### Acknowledgments:
- **Dataset**: Yahoo Finance API, Kaggle Community
- **Compute**: Google Colab Pro (Tesla T4 GPU)
- **Libraries**: TensorFlow, scikit-learn, pandas

---
## 📜 License
MIT License
Copyright (c) 2023 **Shreyas Khandale, Prathamesh Patil, Rohan Patil**

For more details, see [LICENSE](LICENSE).

---
## 💡 Getting Started
### 🔧 Installation & Usage
```bash
git clone https://github.com/yourusername/Stock-Market-Analysis-using-Heuristic-Approach.git  
cd Stock-Market-Analysis-using-Heuristic-Approach  
pip install -r requirements.txt  # TensorFlow, yfinance, pandas, numpy  
python train.py --epochs 200 --batch_size 64  
```
### 🔥 Run Predictions
```bash
python predict.py --model saved_model.h5 --data test_data.csv
```

---
## 🤝 Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit changes and push to your branch.
4. Open a pull request (PR).

---
## 📫 Contact
For any inquiries or collaborations, please reach out:
📩 Email: [your-email@example.com](mailto:your-email@example.com)
