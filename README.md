📌 Tweet Classification

This repository contains the implementation for the script that we use to classify tweets using multiple feature-engineering techniques and machine-learning models.

🔍 Project Overview

The project processes tweet datasets from five domains (UN, Chennai IPL, Shawn Mendes, Man City, NatGeo) and builds a complete ML pipeline including:

Data loading & merging

Bag-of-Words feature generation

Sentence embedding extraction (Transformer-based)

Model training, tuning, and evaluation

Visualization of losses, metrics, and confusion matrices

🧮 Feature Engineering

✔ Bag of Words (BoW)
Custom-built vocabulary extraction and vectorization.

✔ Sentence Embeddings
Semantic vector representations using transformer embeddings.

🤖 Models Implemented

K-Nearest Neighbors (KNN)

Neural Network (MLP Classifier)

Random Forest

Bagging Classifier

Voting Classifier

📊 Evaluation & Analysis

The project includes:

k-fold cross-validation

Training/validation loss plots

Classification reports

Confusion matrix heatmaps

Comparative model analysis on BoW vs Embeddings

Best Performers:

BoW: Voting Classifier

Embeddings: Bagging Classifier

🎯 Goal

To compare traditional and embedding-based features for text classification and analyze how different ML models and ensemble methods perform on real-world tweet data.
