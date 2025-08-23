
# streamlit_app.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

#import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy import sparse

# =========================================================
# 0) Page config
# =========================================================
st.set_page_config(
    page_title="Hybrid Recommender System",
    page_icon="🛒",
    layout="wide"
)

# =========================================================
# 1) Robust Data Loader (NO upload needed)
#    - Tries to load your provided dataset:
#        '/mnt/data/Ecommerce_Delivery_Analytics_New.csv'
#        './Ecommerce_Delivery_Analytics_New.csv'
#    - Falls back to a small in-memory demo dataset so the app ALWAYS runs
# =========================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    possible_paths = [
        "/mnt/data/Ecommerce_Delivery_Analytics_New.csv",   # path in this chat workspace
        "Ecommerce_Delivery_Analytics_New.csv"              # local path if you deploy
    ]
    for p in possible_paths:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8", low_memory=False)
            return df

    # ---- Fallback demo data (ALWAYS AVAILABLE) ----
    demo = pd.DataFrame({
        "order_id": [101,102,103,104,105,106,107,108,109,110],
        "customer_id": ["U1","U2","U3","U1","U2","U4","U5","U3","U1","U2"],
        "product_id": [1,2,3,1,4,5,2,3,5,4],
        "product_name": ["Wireless Earbuds","Smartphone","Laptop","Wireless Earbuds","Smartwatch","Bluetooth Speaker","Smartphone","Laptop","Bluetooth Speaker","Smartwatch"],
        "category": ["Audio","Phones","Computers","Audio","Wearables","Audio","Phones","Computers","Audio","Wearables"],
        "description": [
            "Noise cancelling earbuds with long battery",
            "5G AMOLED screen flagship",
            "Lightweight 16GB RAM laptop",
            "True wireless stereo earbuds",
            "Fitness tracking heart rate",
            "Portable deep bass speaker",
            "High refresh display smartphone",
            "i7 ultrabook performance",
            "Water resistant speaker for travel",
            "GPS calling smartwatch"
        ],
        "price": [2999, 24999, 65999, 2999, 7999, 3499, 24999, 64999, 3499, 7999]
    })
    return demo

df_raw = load_data()

# =========================================================
# 2) Smart column detection (handles many schema variants)
#    We try to auto-detect product/user/category/description/price columns.
# =========================================================
def _find_col(candidates: List[str], df_cols: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df_cols}
    for cand in candidates:
        for col in df_cols:
            if re.fullmatch(cand, col, flags=re.IGNORECASE):
                return col
        for cl, orig in cols_lower.items():
            if re.search(cand, cl, flags=re.IGNORECASE):
                return orig
    return None

def detect_columns(df: pd.DataFrame):
    cols = df.columns.tolist()

    product_col = _find_col(
        [r"(product.*name)", r"(item.*name)", r"product", r"sku", r"item"],
        cols
    ) or _find_col([r"name"], cols)

    user_col = _find_col(
        [r"(customer.*id)", r"(user.*id)", r"(buyer.*id)", r"(client.*id)", r"customer", r"user"],
        cols
    )

    category_col = _find_col(
        [r"category", r"product.*category", r"dept|department|segment|class"],
        cols
    )

    desc_col = _find_col(
        [r"description", r"product.*desc|desc|details|summary|title"],
        cols
    )

    price_col = _find_col(
        [r"price|selling.*price|mrp|list.*price|amount|cost"],
        cols
    )

    order_col = _find_col(
        [r"(order.*id)", r"order"],
        cols
    )

    return {
        "product": product_col,
        "user": user_col,
        "category": category_col,
        "description": desc_col,
        "price": price_col,
        "order": order_col
    }

colmap = detect_columns(df_raw)

# Minimal guardrails: create synthetic columns if missing
df = df_raw.copy()
if colmap["product"] is None:
    df["product_name"] = df.index.astype(str).map(lambda i: f"Item {i}")
    colmap["product"] = "product_name"
if colmap["user"] is None:
    df["customer_id"] = df.groupby(colmap["product"]).cumcount().astype(str).radd("U")
    colmap["user"] = "customer_id"
if colmap["category"] is None:
    df["category"] = "Misc"
    colmap["category"] = "category"
if colmap["description"] is None:
    df["description"] = df[colmap["product"]].astype(str) + " " + df[colmap["category"]].astype(str)
    colmap["description"] = "description"
if colmap["price"] is None:
    df["price"] = np.nan
    colmap["price"] = "price"

# Clean up text
for c in [colmap["product"], colmap["category"], colmap["description"]]:
    df[c] = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

# Deduplicate products on name+category (keep first)
df.drop_duplicates(subset=[colmap["product"], colmap["category"]], keep="first", inplace=True)

# =========================================================
# 3) Popularity (as per guidelines: "Popular Products At Top")
#    - Popularity by frequency of product occurrences in orders/users.
# =========================================================
@st.cache_data(show_spinner=False)
def compute_popularity(raw_df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    # If we have a user column, count distinct users per product; else count rows per product
    if colmap["user"] in raw_df.columns:
        pop = raw_df.groupby(colmap["product"])[colmap["user"]].nunique().rename("popularity").reset_index()
    else:
        pop = raw_df[colmap["product"]].value_counts().rename_axis(colmap["product"]).reset_index(name="popularity")
    return pop.sort_values("popularity", ascending=False)

popularity_df = compute_popularity(df_raw, colmap)

# Join popularity back for convenience
df = df.merge(popularity_df, on=colmap["product"], how="left")
df["popularity"] = df["popularity"].fillna(0)

# =========================================================
# 4) Content-based filtering (TF-IDF on product text)
#    Text = name + category + description (stopwords removed)
# =========================================================
@st.cache_data(show_spinner=False)
def build_content_model(df: pd.DataFrame, colmap: dict):
    text = (
        df[colmap["product"]].astype(str) + " "
        + df[colmap["category"]].astype(str) + " "
        + df[colmap["description"]].astype(str)
    )
    tfidf = TfidfVectorizer(stop_words="english", min_df=1)
    X = tfidf.fit_transform(text)
    return tfidf, X

tfidf, X_tfidf = build_content_model(df, colmap)

@st.cache_data(show_spinner=False)
def content_recommendations(product_name: str, df: pd.DataFrame, colmap: dict, _X) -> pd.DataFrame:
    # Find index of product
    idx = df.index[df[colmap["product"]].str.lower() == product_name.lower()]
    if len(idx) == 0:
        return pd.DataFrame(columns=[colmap["product"], "score"])
    i = idx[0]
    sims = cosine_similarity(_X[i], _X).ravel()
    # Exclude the item itself
    order = np.argsort(-sims)
    order = [j for j in order if j != i]
    top = order[:10]
    out = df.iloc[top][[colmap["product"], colmap["category"], "popularity", colmap["price"], colmap["description"]]].copy()
    out.insert(1, "score", np.round(sims[top], 4))
    return ensure_unique_columns(out)

# =========================================================
# 5) Collaborative filtering (simple, robust, item–item)
#    - Build user–item implicit matrix from co-occurrence
# =========================================================
@st.cache_data(show_spinner=False)
def build_item_item(df_all: pd.DataFrame, colmap: dict):
    # Use unique (user, product) pairs
    tmp = df_all[[colmap["user"], colmap["product"]]].dropna().drop_duplicates()
    # Build index maps
    users = tmp[colmap["user"]].astype(str).unique()
    items = df[colmap["product"]].astype(str).unique()  # ensure aligns with df (after dedupe)
    u_to_i = {u:i for i,u in enumerate(users)}
    p_to_i = {p:i for i,p in enumerate(items)}

    # Build sparse matrix
    rows = tmp[colmap["user"]].astype(str).map(u_to_i).values
    cols = tmp[colmap["product"]].astype(str).map(p_to_i).values
    mask = ~pd.isna(rows) & ~pd.isna(cols)
    rows = rows[mask]
    cols = cols[mask]
    data = np.ones_like(rows, dtype=np.float32)
    UI = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

    # Item-item cosine (normalized co-occurrence)
    # Add small epsilon to avoid div by zero
    item_norms = np.sqrt(UI.power(2).sum(axis=0)).A1 + 1e-9
    sim = (UI.T @ UI).astype(np.float32)
    # Zero diagonal
    sim.setdiag(0)
    # Normalize to cosine-like
    inv_norm = sparse.diags(1.0 / item_norms)
    item_sim = inv_norm @ sim @ inv_norm

    return items, item_sim

items_list, ITEM_SIM = build_item_item(df_raw, colmap)
item_index = {p:i for i,p in enumerate(items_list)}

@st.cache_data(show_spinner=False)
def collaborative_recommendations(product_name: str, df: pd.DataFrame, colmap: dict, items_list: np.ndarray, _item_sim) -> pd.DataFrame:
    if product_name not in item_index:
        # Try case-insensitive
        cand = [p for p in item_index if p.lower() == product_name.lower()]
        if not cand:
            return pd.DataFrame(columns=[colmap["product"], "collab_score"])
        product_name = cand[0]

    i = item_index[product_name]
    sims = _item_sim.getrow(i).toarray().ravel()
    order = np.argsort(-sims)
    order = [j for j in order if j != i]
    top = order[:10]
    rec_names = [items_list[j] for j in top]
    out = df[df[colmap["product"]].isin(rec_names)][[colmap["product"], colmap["category"], "popularity", colmap["price"], colmap["description"]]].drop_duplicates().copy()
    
    out = ensure_unique_columns(out)

    prod_col = out[colmap["product"]]
    if isinstance(prod_col, pd.DataFrame):   # if duplicates snuck through
        prod_col = prod_col.iloc[:, 0]
    # map scores
    score_map = {items_list[j]: float(sims[j]) for j in top}
    
    out.insert(1, "collab_score", out[colmap["product"]].map(score_map).round(4))
    # Order by score then popularity
    out = out.sort_values(["collab_score", "popularity"], ascending=[False, False])
    return out

# =========================================================
# 6) Optional: K-Means clustering over product features
#    - TF-IDF (reduced with SVD) + price
#    - Cluster names from top terms
# =========================================================
#st.write("Current DataFrame columns:", list(df.columns))
#st.write("Current colmap:", colmap)
def sanitize_colmap(df: pd.DataFrame, colmap: dict) -> dict:
    """
    Ensure colmap keys point to unique, existing columns.
    If duplicates exist, pick the first occurrence.
    If missing, drop the mapping.
    """
    new_map = {}
    for key, col in colmap.items():
        if col is None:
            continue
        matches = [c for c in df.columns if c.lower() == col.lower()]
        if not matches:
            continue
        new_map[key] = matches[0]   # pick first occurrence only
    return new_map

@st.cache_data(show_spinner=False)
def kmeans_cluster(df: pd.DataFrame, colmap: dict, _X_text, n_clusters: int = 5):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # ✅ Dedup right away
    df = ensure_unique_columns(df)

    # ---- Reduce text features ----
    n_comp = min(64, _X_text.shape[1]-1) if _X_text.shape[1] > 64 else max(2, _X_text.shape[1]-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Xt = svd.fit_transform(_X_text)
    Xt = np.nan_to_num(Xt, nan=0.0)

    # ---- Add price ----
    price_col = colmap["price"] if colmap["price"] in df.columns else None
    if price_col:
        price = df[price_col].astype(float).fillna(df[price_col].astype(float).median())
    else:
        price = pd.Series([0.0]*len(df), index=df.index)

    scaler = StandardScaler()
    price_scaled = scaler.fit_transform(price.values.reshape(-1,1))

    feats = np.hstack([Xt, price_scaled])
    feats = np.nan_to_num(feats, nan=0.0)

    # ---- KMeans ----
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(feats)

    # ✅ Dedup again before selecting
    df = ensure_unique_columns(df)

    needed = []
    for c in [colmap.get("product"), colmap.get("category"), colmap.get("price"), "popularity", colmap.get("description")]:
        if c and c in df.columns and c not in needed:
            needed.append(c)
    df_clustered = df[needed].copy()
    df_clustered["cluster"] = labels

    # ---- Cluster names ----
    cluster_names = {}
    for c in range(n_clusters):
        subset = df_clustered[df_clustered["cluster"] == c]
        text = (
            subset.get(colmap["product"], pd.Series("", index=subset.index)).astype(str) + " " +
            subset.get(colmap["category"], pd.Series("", index=subset.index)).astype(str) + " " +
            subset.get(colmap["description"], pd.Series("", index=subset.index)).astype(str)
        )
        words = text.str.lower().str.findall(r"[a-z]{3,}")
        all_words = pd.Series([w for lst in words.tolist() for w in lst])
        cluster_names[c] = ", ".join(all_words.value_counts().head(3).index.tolist()) if not all_words.empty else f"Cluster {c}"

    df_clustered["cluster_name"] = df_clustered["cluster"].map(cluster_names)

    # ✅ Final dedup before return
    return ensure_unique_columns(df_clustered), cluster_names







# =========================================================
# 7) UI
# =========================================================
st.title("🛒 Hybrid Recommender System")
st.caption(
    "Popular ➜ Content-Based ➜ Collaborative. "
    "No CSV upload needed — works out of the box. "
    "If your dataset file is present, it will be used automatically."
)

# Sidebar controls
st.sidebar.header("Controls")
top_k_pop = st.sidebar.slider("How many Popular products to show?", 5, 50, 15, 1)
do_cluster = st.sidebar.checkbox("Enable K-Means Clustering (exploration)", value=True)
k_clusters = st.sidebar.slider("Number of clusters", 3, 12, 5, 1, disabled=not do_cluster)

# Popular products
# Popular products
st.subheader("🔥 Popular Products")
pop_view = df[[colmap["product"], colmap["category"], "popularity", colmap["price"]]].drop_duplicates()
pop_view = pop_view.sort_values("popularity", ascending=False).head(top_k_pop)

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataframe has strictly unique column names.
    If duplicates exist, suffixes _1, _2... are added.
    """
    df = df.copy()
    new_cols = []
    seen = {}
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
    df.columns = new_cols
    return df


# Fix duplicate cols once here
pop_view = ensure_unique_columns(pop_view)


st.dataframe(pop_view, use_container_width=True)


# Select product
st.subheader("🎯 Choose a product to get recommendations")
product_choices = pop_view[colmap["product"]].tolist()  # popular first
# Add additional products (not in top list) to the dropdown
other_products = df[~df[colmap["product"]].isin(product_choices)][colmap["product"]].tolist()
product_choices += other_products
selected_product = st.selectbox("Product", product_choices)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📚 Content-Based Recommendations")
    c_recs = content_recommendations(selected_product, df, colmap, X_tfidf)

    # ✅ Ensure unique before display
    c_recs = ensure_unique_columns(c_recs)

    if c_recs.empty:
        st.info("No content-based matches found.")
    else:
        st.dataframe(c_recs.head(10), use_container_width=True)


with col2:
    st.markdown("#### 👥 Collaborative (Item–Item) Recommendations")
    i_recs = collaborative_recommendations(selected_product, df, colmap, items_list, ITEM_SIM)

    # ✅ Ensure unique before display
    i_recs = ensure_unique_columns(i_recs)

    if i_recs.empty:
        st.info("Not enough user–item co-occurrence to compute collaborative recs.")
    else:
        st.dataframe(i_recs.head(10), use_container_width=True)


# Clustering section
if do_cluster:
    st.subheader("🧩 Product Clusters (K-Means)")

    try:
        df_clustered, cluster_names = kmeans_cluster(df, colmap, X_tfidf, n_clusters=k_clusters)

        cluster_sel = st.selectbox(
            "View products in cluster",
            sorted(df_clustered["cluster"].unique().tolist()),
            format_func=lambda c: f"{c} — {cluster_names.get(c, f'Cluster {c}')}"
        )

        cluster_view = df_clustered[df_clustered["cluster"] == cluster_sel] \
            .sort_values("popularity", ascending=False) \
            .head(50)

        st.dataframe(ensure_unique_columns(cluster_view), use_container_width=True)

    except Exception as e:
        st.warning(f"Clustering step skipped due to: {e}")


# =========================================================
# 8) Extra: Simple hybrid list for the selected product
#    - Merge content & collaborative with popularity re-ranking
# =========================================================
st.subheader("🔀 Hybrid Recommendations (Re-ranked)")
def hybrid_rank(product_name: str, df: pd.DataFrame, colmap: dict, _X, items_list: np.ndarray, _item_sim,
                w_content=0.6, w_collab=0.4) -> pd.DataFrame:
    """Combine content & collaborative scores with popularity boost."""
    c = content_recommendations(product_name, df, colmap, _X).rename(columns={"content_score": "content_score"})
    k = collaborative_recommendations(product_name, df, colmap, items_list, _item_sim).rename(columns={"collab_score": "collab_score"})

    merged = pd.merge(
        c, k,
        on=[colmap["product"], colmap["category"], "price", "popularity", colmap["description"]],
        how="outer"
    )
    merged = ensure_unique_columns(merged)

    # ✅ fix: handle missing cols safely
    if "content_score" not in merged.columns:
        merged["content_score"] = 0.0
    else:
        merged["content_score"] = merged["content_score"].fillna(0.0)

    if "collab_score" not in merged.columns:
        merged["collab_score"] = 0.0
    else:
        merged["collab_score"] = merged["collab_score"].fillna(0.0)

    pop_boost = np.log1p(merged["popularity"].fillna(0))
    merged["hybrid_score"] = w_content * merged["content_score"] + w_collab * merged["collab_score"] + 0.05 * pop_boost

    merged = merged[merged[colmap["product"]].str.lower() != product_name.lower()]
    return merged.sort_values(["hybrid_score", "popularity"], ascending=[False, False])



hyb = hybrid_rank(selected_product, df, colmap, X_tfidf, items_list, ITEM_SIM)

# ✅ Ensure unique before display
#hyb = ensure_unique_columns(hyb)

if hyb.empty:
    st.info("Hybrid list is empty for this selection.")
else:
    st.dataframe(hyb.head(15), use_container_width=True)

# =========================================================
# 9) Footer: explainability
# =========================================================
with st.expander("ℹ️ What this app does (quick summary)"):
    st.markdown("""
- **Popular Products**: ranked by distinct users (or frequency) per product.
- **Content-Based**: TF-IDF on `name + category + description`; cosine similarity finds similar items.
- **Collaborative**: item–item similarity from user–item co-occurrence (implicit feedback).
- **Hybrid**: combines content & collaborative, with a small popularity boost so **popular items float to the top**.
- **K-Means**: clusters products (SVD-reduced text + price) to explore groups and themes.
    """)
