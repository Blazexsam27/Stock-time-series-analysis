# 📈 Multi-Stock Price Prediction using LSTM (PyTorch)

This project predicts future stock prices using **Long Short-Term Memory (LSTM)** networks trained on **7+ years of stock data** from multiple companies (META, GOOGL, AAPL, AMZN, MSFT, NVDA, TSLA, and more).  
It demonstrates the use of **multi-stock embeddings**, **sequence modeling**, and **deep learning-based time series forecasting**.

---

## 🧠 Key Features

- 🔹 **Multi-stock LSTM architecture** — a single model learns relationships across 10 stocks using *stock embeddings*.
- 🔹 **Sliding window sequence modeling** — predicts the next closing price from past 60 days.
- 🔹 **Cross-stock generalization** — reduces overfitting by learning shared patterns.
- 🔹 **Evaluation metrics:** RMSE and R² (per stock + overall).
- 🔹 **Beautiful performance visualizations** (actual vs predicted, error histograms, R² summary, etc.).

---

## 📊 Model Architecture

```python
class MultiStockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, num_stocks=10, embed_dim=4):
        super(MultiStockLSTM, self).__init__()
        self.stock_embed = nn.Embedding(num_stocks, embed_dim)
        self.lstm = nn.LSTM(input_size + embed_dim, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, stock_id):
        stock_emb = self.stock_embed(stock_id)
        stock_emb = stock_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, stock_emb), dim=2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```
### 🧩 Dataset

| Feature    | Description                                    |
| ---------- | ---------------------------------------------- |
| Open       | Opening price                                  |
| High       | Daily high                                     |
| Low        | Daily low                                      |
| Close      | Closing price                                  |
| Volume     | Trading volume                                 |
| (Added)    | Technical indicators like RSI, EMA, MACD, etc. |

### ⚙️ Training Configuration

| Parameter       | Value           |
| --------------- | --------------- |
| Sequence length | 60 days         |
| Hidden size     | 64              |
| LSTM layers     | 2               |
| Dropout         | 0.3–0.4         |
| Batch size      | 64              |
| Optimizer       | Adam (lr=0.001) |
| Loss function   | MSELoss         |

### 📈 Performance

| Dataset    | RMSE | R²   |
| ---------- | ---- | ---- |
| Validation | 0.04 | 0.99 |
| Test       | 1.12 | 0.67 |


### 🧰 Tech Stack

Language: Python
Libraries: PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
Environment: Google Colab / Jupyter Notebook
Version Control: Git + GitHub
