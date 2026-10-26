# Product Category Classification - Machine Learning Pipeline

**Author:** Gligor Milićević  
**Course:** Machine Learning & Data Engineering (IT Academy)  

---

## 📂 Project Overview
This project implements an end-to-end supervised machine learning pipeline for automated **multi-class product category classification**. Using product title text features extracted from e-commerce datasets, the system cleans data, engineers statistical features, vectorizes text using **TF-IDF**, and classifies products using a **Logistic Regression** classifier.

---

## 🗂️ Repository Directory Structure
```text
Gligor_Milicevic_Task3/
│
├── data/
│   └── products.csv             # Raw dataset
├── notebooks/
│   └── product_category_analysis.ipynb # Exploratory data analysis & model training notebook
├── src/
│   ├── train_model.py           # Automated training and serialization script
│   └── predict_category.py      # Inference script for new product titles
├── models/
│   ├── product_classifier_model.pkl # Serialized Logistic Regression model
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF text vectorizer
└── README.md                    # Project documentation