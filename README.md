# E-Commerce Customer Behaviour Analysis

Exploratory data analysis and machine learning project investigating customer purchasing patterns, platform engagement, and revenue drivers from a Turkish e-commerce platform dataset.

---

## Overview

This project analyses transactional, demographic, and behavioural data from 5,000 customer orders. The goal is to extract actionable business insights through a full end-to-end analytics workflow — from data quality validation through to prioritised strategic recommendations.

A central theme emerges across the analysis: **revenue on this platform is driven by product mix and purchase quantity, not by customer demographics or browsing behaviour.** This finding shapes the segmentation approach, the modelling decisions, and every recommendation in the final section.

The analysis is structured around four business questions:

1. **Who are the customers, and how do they interact with the platform?**
2. **Which factors most strongly influence customer spending?**
3. **Can customers be segmented into meaningful behavioural groups?**
4. **Which features best predict order value?**

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — E-Commerce Customer Behavior and Sales Analysis (TR)](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr) |
| Records | 5,000 customer orders |
| Features | 18 variables |
| Types | Transactional, demographic, behavioural, operational |
| Key variables | Order value, product category, quantity, session duration, page views, customer rating, delivery time |
| Target variable | `total_amount` (used in predictive modelling) |
| Currency | Turkish Lira (TRY, synthetic) |

> Note: this is a synthetic dataset used for learning and demonstration purposes. All findings are illustrative rather than drawn from real business data.

---

## Tools & libraries

| Tool | Purpose |
|---|---|
| Python | Core analysis language |
| Pandas | Data manipulation and aggregation |
| NumPy | Numerical operations |
| Plotly | Interactive visualisations |
| Scikit-Learn | Preprocessing, clustering, regression pipelines |
| Google Colab | Development environment |

---

## Project structure

### 1. Data quality & validation
- Shape, structure, and summary statistics
- Missing value and duplicate detection
- Categorical consistency checks
- **Plausibility checks:** non-positive transactional values, and reconciliation of `total_amount` against `unit_price × qty − discount_amount`
- **Finding:** the dataset is clean, internally consistent, and suitable for analysis

### 2. Customer & platform overview
- Demographic distribution (age groups, gender, city)
- Platform engagement (device type, session duration, page views)
- Transactional patterns (product categories, payment methods, delivery time)
- **Key finding:** the core demographic is aged 26–45; mobile accounts for ~56% of sessions; product category order volumes appear balanced on the surface, but revenue tells a different story

### 3. Drivers of purchasing behaviour
- Demographics vs spending — age groups, gender, city volume-value tradeoff
- Engagement vs order value — session duration and page views (density heatmaps, correlation matrix)
- Operational metrics — delivery time vs rating, returning vs new customer behaviour
- Product category vs order value
- **Key finding:** product mix is the primary driver of revenue; demographics, engagement, and operational metrics show limited explanatory power; Electronics generates a disproportionate share of revenue despite similar order volumes to other categories

### 4. Advanced analytics

#### 4.1 Customer segmentation — K-Means clustering
Optimal k selected using both the elbow method and silhouette analysis (k=3). Pipeline: StandardScaler + OneHotEncoder → KMeans. Cluster profiles visualised via PCA projection.

| | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| **Label** | High-Value Buyers | Low-Value Food Buyers | Occasional Buyers |
| **Share of customers** | 24.2% | 23.1% | 52.8% |
| **Share of revenue** | 59.9% | 13.9% | 26.2% |
| **Avg. order value** | ~2,439 | ~590 | ~488 |
| **Avg. qty / order** | ~4.2 | ~1.8 | ~1.5 |
| **Avg. rating** | 4.10 | 2.29 | 4.52 |
| **Key category** | Electronics | Food | Books |

#### 4.2 Revenue prediction — Random Forest regression
`unit_price` and `discount_amount` excluded to avoid target leakage. `max_depth=10` applied to limit overfitting.

| Metric | Value |
|---|---|
| R² | 0.59 |
| RMSE | 1,295 |

**Top predictors:**
- Electronics category — importance 0.316
- Quantity — importance 0.259
- Age, session duration, page views — modest secondary contribution (~0.05–0.06 each)
- Gender, city, device, payment method — negligible predictive power

### 5. Strategic recommendations
Five prioritised recommendations grounded in specific findings, each traceable to a section and metric:

| Priority | Focus | Key lever |
|---|---|---|
| 1 | Retain Cluster 0 | Loyalty programme, upsell, payment flexibility |
| 2 | Activate Cluster 2 | Category migration, bundling, re-engagement |
| 3 | Recover Cluster 1 | Investigate low satisfaction (avg 2.29), then cross-sell |
| 4 | Product portfolio | Invest in high-value categories; reposition low-margin ones |
| 5 | Redirect spend | Deprioritise delivery optimisation and demographic targeting |

---

## How to run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) or any Jupyter-compatible environment
2. The dataset loads automatically via `kagglehub` — no manual download needed:
   ```python
   import kagglehub
   from kagglehub import KaggleDatasetAdapter
   df = kagglehub.dataset_load(
       KaggleDatasetAdapter.PANDAS,
       "umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr",
       "ecommerce_customer_behavior_dataset.csv"
   )
   ```
3. If running locally, install dependencies:
   ```bash
   pip install pandas numpy plotly scikit-learn kagglehub
   ```
4. Run all cells in order: **Runtime → Run all**

---

## Status

Complete. All five sections are finalised including cluster interpretation, predictive modelling results, residual analysis, and strategic recommendations.

---

## Author

**Diogo Silva**  
Data Analyst  
[LinkedIn](https://linkedin.com/in/diogo-silva-612774356) · [Portfolio](https://DiogoAzevedoSilva.github.io)
