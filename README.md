# 🏠 Bangalore Rental Fairness Analyzer

> *"Am I being overcharged for my rent?"* — This project answers that question using data.

## 📌 Project Overview

A data science project that analyzes Bangalore rental listings to detect overpriced properties and generate a **Fairness Score** for any locality. Built as a personal tool to evaluate rental options across key Bangalore neighborhoods.

**Personally motivated** — built this while actively searching for a flat in Bangalore. Wanted data to back my decisions, not just gut feeling.

---

## 🔍 What It Does

- **Fairness Score (0–100)** — tells you if a listing is priced fairly vs the market
- **Overpriced Detection** — flags listings statistically above their locality average using Z-score analysis
- **Fair Rent Prediction** — Linear Regression model predicts what a fair rent should be based on BHK type and locality
- **Geospatial Map** — interactive Folium map of Bangalore color-coded by fairness score
- **Locality Rankings** — ranks 218 Bangalore localities by value-for-money
- **Interactive Dashboard** — Streamlit app to check any listing instantly

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Data processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Folium |
| Machine Learning | Scikit-learn (Linear Regression) |
| Anomaly Detection | Z-score (SciPy) |
| Dashboard | Streamlit |
| Data Source | Kaggle — Bangalore House Rent Dataset |

---

## 📊 Key Findings

- **BTM Layout, JP Nagar, Electronics City** — best value areas, scoring 98+/100
- **Richmond Town, Domlur, Indiranagar** — most overpriced, scoring 15–48/100
- **157 listings** flagged as overpriced across 218 localities
- Model achieves **R² = 0.52** with Mean Absolute Error of ₹5,069

---

## 🗂️ Project Structure

```
Bangalore-Rental-Analyzer/
├── data/
│   ├── BangaloreHouseRentDtls.csv
│   ├── BangaloreRent.csv
│   ├── bangalore_rent_scored.csv
│   └── locality_ranking.csv
├── notebooks/
│   └── 01_data_exploration.ipynb
├── outputs/
│   ├── 01_locality_rent_comparison.png
│   ├── 02_fairness_scores.png
│   ├── 03_locality_ranking.png
│   └── bangalore_rental_map.html
├── app.py
└── README.md
```

---

## 🚀 How to Run

```
pip install pandas numpy matplotlib seaborn scikit-learn folium streamlit streamlit-folium scipy
python -m streamlit run app.py
```

---

## 📈 Fairness Score Formula

```
Fairness Score = 100 - max(0, % above predicted rent) - (Z-score penalty x 5)
```

- **80–100** ✅ Fair Price
- **60–79** ⚠️ Slightly High
- **40–59** 🔶 Overpriced
- **0–39** 🚨 Significantly Overpriced

---

## 👩‍💻 Author

**Yayavari R** — Systems Engineer at Infosys, transitioning to Data/Business Analyst roles.

[GitHub](https://github.com/Yayavari) · [LinkedIn](https://linkedin.com/in/yayavari-r)