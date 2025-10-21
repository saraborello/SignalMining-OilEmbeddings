
# Semantic Signal Mining for Oil Price Forecasting Using News Embeddings
This project investigates the integration of textual data ex- tracted from market news with natural language processing techniques to enhance crude oil price forecasting. Through au- tomated scraping of relevant articles and reports, up-to-date textual data were collected and transformed into numerical rep- resentations using embeddings. These embeddings were then used as inputs for machine learning models to capture hidden informational signals within the news and improve the accuracy of oil price predictions. The study demonstrates how semantic analysis of financial news can effectively complement historical market data, providing an innovative and data-driven approach to forecasting in the energy sector.


## Key Features

* **Multi-Source Data Collection** – Automated web scraping from *OilPrice.com*.
* **NLP Modeling** – Integration of Word2Vec, DistilBERT, and Sentence Transformers.
* **Sentiment Analysis** – Daily sentiment scoring based on market news.
* **Time Series Forecasting** – Price prediction using LSTM and Random Forest.
* **Semantic Analysis** – Cosine similarity, drift detection, and dispersion tracking.


## Data Sources

| Source               | Description                                                |
| :------------------- | :--------------------------------------------------------- |
| **Yahoo Finance**    | Brent crude oil futures *(BZ=F)*                           |
| **OilPrice.com**     | News articles with metadata (title, author, date, excerpt) |
| **Sentiment Scores** | Aggregated daily sentiment indicators                      |
| **Embeddings**       | Pre-computed vectors from multiple NLP models              |

## Repository Structure

```
SignalMining-OilEmbeddings/
├── Data/
│   ├── Data Retrieval/          # Web scraping notebooks
│   ├── Models/                  # Model configurations
│   └── raw/                     # Raw datasets
│       ├── daily_sent.csv
│       ├── df_oilnews.csv
│       ├── embeddings_*.csv
│       └── ...
├── Models/                      # Embedding and model development
│   ├── embeddings_*.ipynb
│   ├── Sentiment.ipynb
│   └── ...
├── Brent/                       # Brent oil price analysis
│   ├── Brent_model.ipynb
│   ├── best_lstm.pt
│   └── ...
└── README.md
```

---

## Getting Started

### Prerequisites

* Python **3.8+**
* Jupyter Notebook
* Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SignalMining-OilEmbeddings.git
cd SignalMining-OilEmbeddings

# 2. Install dependencies
pip install pandas numpy scikit-learn plotly yfinance requests beautifulsoup4 torch transformers sentence-transformers

# 3. Launch Jupyter Notebook
jupyter notebook
```

---

## Main Notebooks

| Notebook                                            | Description                                           |
| --------------------------------------------------- | ----------------------------------------------------- |
| **Data/Data Retrieval/Oilnews.ipynb**               | Web scraping of oil-related articles                  |
| **Models/embeddings_word2vec.ipynb**                | Word2Vec embeddings                                   |
| **Models/embeddings_distilbert-base-uncased.ipynb** | DistilBERT contextual embeddings                      |
| **Models/embeddings_all-MiniLM-L6-v2.ipynb**        | Sentence Transformer embeddings                       |
| **Brent/Brent_model.ipynb**                         | Forecasting model using Random Forest                 |
| **Brent/transformers_autoencoders.ipynb**           | Transformer-based autoencoders for feature extraction |

---

## Methodology

### 1. Data Collection

* Automated retrieval of oil market news.
* Sentiment computation via NLP pipelines.
* Integration with historical Brent price data.

### 2. Feature Engineering

* **Semantic Embeddings**: Vectorized representation of textual meaning.
* **Drift Detection**: Monitoring linguistic shifts over time.
* **Cosine Similarity**: Semantic distance computation between news items.
* **Aggregated Sentiment**: Daily indicators derived from multiple articles.

### 3. Model Architecture

* **LSTM Networks** – Sequential models for temporal dependencies.
* **Random Forest** – Ensemble learning for regression tasks.
* **Autoencoders** – Dimensionality reduction and latent feature discovery.

### 4. Evaluation Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² Score
* Mean Absolute Percentage Error (MAPE)

---

## Analytical Components

### Embedding Techniques

| Model                     | Description                                             |
| :------------------------ | :------------------------------------------------------ |
| **Word2Vec**              | Static word embeddings capturing lexical co-occurrence. |
| **DistilBERT**            | Transformer-based contextual embeddings.                |
| **Sentence Transformers** | Sentence-level semantic representations.                |

### Advanced Analysis

* **Semantic Drift** – Evolution of thematic content over time.
* **Cosine Similarity** – Quantitative semantic proximity.
* **Dimensionality Reduction** – 2D and 3D visualization of semantic space.


---

## Results

The project demonstrates the effectiveness of combining **sentiment analysis**, **semantic embeddings**, and **time series forecasting** for modeling oil market behavior.
Hybrid architectures incorporating NLP features significantly improve predictive accuracy over purely numerical baselines.
