# ADHD-ML-Analyzer

ADHD-ML-Analyzer is a Python-based machine learning project designed to analyze mental health data and identify patterns related to ADHD and other psychological conditions. Using data preprocessing, clustering, and classification techniques, this tool helps visualize mental health patterns and predict potential mental health issues.

---

## Features

- **Data Cleaning & Preprocessing**: Handles missing values, categorical conversion, and standardizes survey responses.
- **Clustering**: Uses PCA for dimensionality reduction and KMeans to identify clusters of participants based on mental health metrics.
- **Classification**: Uses LazyPredict to benchmark multiple models and selects the best performing algorithm (KNN) for prediction.
- **Visualization**: Scatterplots for clusters with silhouette scores to evaluate clustering quality.
- **Diagnosis Analysis**: Identifies the presence of mental health conditions, including ADHD, anxiety, depression, bipolar disorder, and more.

---

## Dataset

The project uses a dataset `ADHD.xlsx` containing survey responses on:

- Mental health history
- Therapy and medication usage
- ADHD-related symptoms
- Demographics and test scores

### Preprocessing steps include:

- Removing irrelevant columns such as age, sex, and language.
- Imputing missing values with the most frequent values.
- Encoding categorical variables (yes/no) as binary.
- Creating new features to track both self-reported and diagnosed conditions.
- Generating a combined target variable `being ill` for mental health status.

## Languge & Tools:
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![LazyPredict](https://img.shields.io/badge/LazyPredict-FF7F50?style=for-the-badge&logo=python&logoColor=white)
![OpenPyXL](https://img.shields.io/badge/OpenPyXL-4B8BBE?style=for-the-badge&logo=python&logoColor=white)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/raha1382/ADHD-ML-Analyzer.git
cd ADHD-ML-Analyzer
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
##Results

Clustering: Two primary clusters identified with centroids:
[[ 30.32533231  0.9326343 ]
 [-19.82047863 -0.6095649 ]]
 Classification: KNN achieves high accuracy and F1 scores, effectively predicting the being ill status based on survey data.

Scatterplots provide a visual representation of clusters and their distribution.

The analysis demonstrates the potential of machine learning in identifying mental health issues from survey data. KNN provides the most reliable classification results in this dataset.
