# Hybrid-recommender
# 🛒 Hybrid Recommender System

This is a **Streamlit app** that demonstrates a **Hybrid Product Recommender System** for e-commerce analytics.  
It combines **Popularity-based**, **Content-based**, and **Collaborative Filtering** approaches, and optionally clusters products using **K-Means**.

---

## 🚀 Features
- **🔥 Popular Products** → ranked by distinct users/orders per product  
- **📚 Content-Based Filtering** → TF-IDF on product name + category + description, cosine similarity  
- **👥 Collaborative Filtering** → item–item co-occurrence (implicit user–item matrix)  
- **🔀 Hybrid Recommender** → re-ranks by combining content & collaborative scores + popularity boost  
- **🧩 K-Means Clustering** → clusters products using text embeddings + price for exploration  

---

## 📂 Dataset
- The app automatically tries to load your dataset:
  - `Ecommerce_Delivery_Analytics_New.csv` (if present in repo or workspace)  
- If no dataset is found, it **falls back to a small built-in demo dataset** (so the app always runs).  

---

## ⚡ Running Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/hybrid-recommender.git
   cd hybrid-recommender
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run Streamlit app:
   ```bash
   streamlit run Ecom_app.py

## streamlit deployed app link:
https://hybrid-recommender-vwcxxhjutnz2zzp9ebgyxq.streamlit.app/
