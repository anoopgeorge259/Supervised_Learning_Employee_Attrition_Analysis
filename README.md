# Employee Attrition Prediction — Supervised Machine Learning

A supervised machine learning project that predicts whether an employee is likely to
leave an organization, and identifies the key factors driving attrition, using an
HR analytics dataset of 10,000 employees.

## 📌 Problem Statement

A leading IT services company has seen rising employee resignations over the past two
years. HR currently relies on manual analysis to guess who might leave — a slow and
inaccurate process. This project builds a data-driven, AI-based solution to proactively
flag employees at risk of attrition, so HR can intervene early, understand the root
causes, and design better retention strategies.

**Type:** Supervised Machine Learning – Binary Classification
**Target variable:** `Attrition` (`Yes` = left the company, `No` = stayed)

## 📂 Dataset

- **File:** `employee_attrition_dataset_10000.csv`
- **Rows:** 10,000 employees
- **Columns:** 26, including Age, Gender, Department, Job Role, Monthly Income,
  Overtime, Job Satisfaction, Work-Life Balance, Performance Rating, Years at Company,
  Absenteeism, and more.
- **Class balance:** ~80% stayed (`No`) vs ~20% left (`Yes`) — an imbalanced dataset.

## 🔍 Project Workflow

1. **Data Understanding** — shape, datatypes, missing values, duplicates, statistical summary
2. **Exploratory Data Analysis (EDA)** — attrition distribution, attrition by department,
   overtime, income, and job satisfaction, plus a correlation heatmap of numeric features
3. **Data Preprocessing** — dropping identifiers, missing-value and outlier checks,
   one-hot encoding of categorical features, stratified train-test split, feature scaling
4. **Model Building** — three models trained and compared:
   - Decision Tree (`class_weight="balanced"`)
   - Random Forest
   - Logistic Regression
5. **Model Evaluation** — accuracy, precision, recall, F1-score, and confusion matrices,
   with particular attention to recall on the minority (`Attrition = Yes`) class
6. **Risk Scoring** — using the Random Forest's predicted probabilities to bucket
   employees into Low / Medium / High attrition-risk tiers
7. **Business Insights & Recommendations** — key drivers of attrition, high-risk
   employee profiles, at-risk departments, and suggested HR interventions

## 📊 Key Findings

- The dataset is imbalanced (~80/20), so **accuracy alone is misleading** — Random
  Forest and Logistic Regression reach ~80% accuracy but achieve **0% recall** on
  employees who actually leave, since they simply predict the majority class.
- The **Decision Tree** (with balanced class weights) recovers ~53% recall on leavers,
  making it the more operationally useful model here, despite lower precision.
- The strongest predictors of attrition (via Random Forest feature importance) are
  **Monthly Income, Hourly Rate, Training Hours, Distance From Home, and Age** —
  compensation and workload factors outweigh satisfaction scores in this dataset.
- Department and overtime show only weak differences in attrition rate on their own.

## 🛠️ Tech Stack

- Python 3
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — preprocessing, modeling, evaluation

## 🚀 How to Run

1. Clone this repository.
2. Open `Employee_Attrition_Analysis_MyDataset.ipynb` in Google Colab or Jupyter.
3. If using Colab, mount your Google Drive and make sure
   `employee_attrition_dataset_10000.csv` is uploaded, then update the file path in the
   "loading dataset" cell if needed.
4. Run all cells top to bottom.

## 📁 Repository Structure

```
├── Employee_Attrition_Analysis_MyDataset.ipynb   # Main analysis notebook
├── employee_attrition_dataset_10000.csv          # Dataset (10,000 employee records)
└── README.md                                     # Project documentation
```

## 📈 Future Improvements

- Address class imbalance with SMOTE, class-weighted Random Forest/Logistic Regression,
  or decision-threshold tuning to improve recall without sacrificing usability.
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV) across all three models.
- Try gradient-boosted models (XGBoost, LightGBM) for comparison.
- Deploy the best model behind a simple API or dashboard for HR to score employees live.

## 🎓 Acknowledgment

This project was completed as part of the **Advanced AI — Supervised Learning**
assignment (IIT Palakkad / ICT Academy Kerala).
