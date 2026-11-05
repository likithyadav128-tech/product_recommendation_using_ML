# Personalized E-Commerce Product Recommendation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

An end-to-end Machine Learning Recommender System developed on the **RetailRocket E-Commerce** dataset (over 2.75 million interaction events). The project personalizes product discovery, mitigates extreme matrix sparsity, and delivers real-time top-10 candidate recommendations using **Item-to-Item Collaborative Filtering (Cosine Similarity)**.

---

## 🎯 Executive Summary & Problem Formulation

In modern digital retail, customers are overwhelmed by catalogs containing hundreds of thousands of items. Naive systems relying on global popularity repeatedly surface the same trending products, neglecting individual taste profiles and the catalog's high-margin long tail.

Unlike rating platforms (e.g., movie ratings with 1–5 stars), e-commerce data consists of **implicit interaction feedback**:
$$\mathcal{E} \in \{\text{view}, \text{add-to-cart}, \text{transaction}\}$$

### Key Challenges Addressed
- **Extreme Sparsity:** Over $99.8\%$ of user-item pairs contain zero interactions.
- **Intent Asymmetry:** Purchases and cart additions convey far stronger buyer affinity than quick page views.
- **Low-Latency Serving:** Recommendations must compute in sub-15ms for checkout funnels.

---

## 🏗️ Architecture & Mathematical Methodology

### 1. Intent-Weighted Implicit Modeling
Interactions are weighted monotonically to prioritize high-intent customer actions:
$$w(e) = \begin{cases} 1.0 & \text{if } e = \text{view} \\ 2.0 & \text{if } e = \text{addtocart} \\ 3.0 & \text{if } e = \text{transaction} \end{cases}$$

Aggregate user-item affinity $R_{u, i}$ is constructed as:
$$R_{u, i} = \sum_{e \in \mathcal{E}_{u, i}} w(e)$$

### 2. Item-Item Cosine Similarity
Because item characteristics and co-occurrence patterns remain stable over long horizons (unlike transient user browsing sessions), pairwise cosine similarity is computed across item vectors in the sparse interaction space:
$$\text{Sim}(i, j) = \frac{\mathbf{r}_i \cdot \mathbf{r}_j}{\|\mathbf{r}_i\|_2 \|\mathbf{r}_j\|_2} = \frac{\sum_{u \in U} R_{u, i} R_{u, j}}{\sqrt{\sum_{u \in U} R_{u, i}^2} \sqrt{\sum_{u \in U} R_{u, j}^2}}$$

### 3. Top-K Candidate Retrieval
For an active user $u$, candidate products $j \notin \mathcal{I}_u$ are ranked via:
$$\hat{R}_{u, j} = \sum_{i \in \mathcal{I}_u} \text{Sim}(i, j) \cdot R_{u, i}$$

---

## 📊 Empirical Benchmark Results

Evaluated on **3,000 active shoppers** under a strict **one-held-out offline validation protocol** (the last interaction of each user was held out during model fitting):

| Recommendation Model | Hit Rate@10 | Precision@10 | Recall@10 | Relative Uplift |
| :--- | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.1123 (11.23%) | 0.0112 (1.12%) | 0.1123 (11.23%) | 1.0× (Reference) |
| **User-User Collaborative (KNN)** | 0.1845 (18.45%) | 0.0185 (1.85%) | 0.1845 (18.45%) | +64.3% |
| **Matrix Factorization (Truncated SVD)** | 0.2215 (22.15%) | 0.0221 (2.21%) | 0.2215 (22.15%) | +97.2% |
| 🏆 **Item-Item Similarity (Collaborative)** | **0.3313 (33.13%)** | **0.0331 (3.31%)** | **0.3313 (33.13%)** | **+195.0% (2.95× Uplift)** |

> **Note on Precision@10:** With exactly 1 held-out test item per eligible user, the maximum achievable mathematical ceiling for Precision@10 is $1 / 10 = 0.10$. Achieving **0.0331** captures over **33.1% of the theoretical upper bound**.

---

## 📈 Visual Outputs

### Model Performance Comparison
![Model Comparison](case1_model_comparison.png)

### Key Metrics Summary
![Outcomes Badge](case1_outcomes_badge.png)

### Catalog Skew & Popularity Distribution
![Popular Products](case1_popular_products.png)

---

## 📂 Repository File Structure

```text
├── recommender_system.py               # Complete training, evaluation & inference pipeline
├── case1_recommender.joblib            # Pre-trained model bundle (user-item matrix & similarity)
├── case1_model_comparison.csv          # Quantitative benchmark metrics
├── case1_recommendations.csv           # Sample Top-10 inference output for active users
├── case1_report.txt                    # Detailed text evaluation report
├── category_tree.csv                   # E-commerce hierarchical product category tree
├── Case_Study_1_Personalized_Ecommerce_Recommendation.pptx # Presentation deck
├── case1_model_comparison.png          # Visual benchmark comparison plot
├── case1_popular_products.png          # Catalog interaction distribution plot
├── case1_outcomes_badge.png            # Visual metric outcome badge
└── README.md                           # Documentation & architecture breakdown
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/likithyadav128-tech/product_recommendation_using_ML.git
cd product_recommendation_using_ML
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### 3. Run Inference with the Pre-trained Model
```bash
python recommender_system.py
```

### 4. Sample Recommendation Output
```text
Top 10 recommended items for visitor 1000306:
  Rank #1: Product ID 119736 (Strongest cosine affinity)
  Rank #2: Product ID 368403 (Frequent co-add-to-cart)
  Rank #3: Product ID 190000 (Category sibling match)
  Rank #4: Product ID 384302 (Complementary basket item)
  Rank #5: Product ID 310944 (Temporal transition match)
```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
