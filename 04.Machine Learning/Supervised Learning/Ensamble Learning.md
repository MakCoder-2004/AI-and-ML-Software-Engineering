# Ensemble Learning

**Ensemble Learning Algorithms:** Bagging, Boosting & Random Forest
Eng. Kholoud Amer

---

## What is Ensemble Learning

**Definition:**
Ensemble learning combines multiple models (often called **"weak learners"**) to produce a better performing **"strong learner"**.

**Why use Ensemble Learning?**

* Reduces errors
* Improves accuracy
* Increases robustness


## Types of Ensemble Methods

* **Bagging (Bootstrap Aggregating)**
* **Boosting**
* **Random Forest** (uses Bagging + Decision Trees)

---

## Bagging

**What is Bagging?**

* Bagging = Bootstrap Aggregating
* Create many random samples from the training data (with replacement)
* Train a separate model on each sample
* Combine their outputs (e.g., by voting for classification or averaging for regression)
* **Goal:** Reduce variance (make model more stable)

### Bagging — Real-World Applications

* **Retail:** Forecasting customer demand (predicting sales using regression)
* **Healthcare:** Predicting patient readmission risks
* **Banking:** Detecting anomalies in transaction patterns

**Often used with:**

* Decision Trees
* KNN
* SVM (less common)

---

## Random Forest: Built on Bagging

* Random Forest = Bagging + Decision Trees
* Creates many decision trees
* Each tree gets a random sample of data (bagging)
* Each tree also uses a random subset of features
* **Final prediction:** majority vote (classification) or average (regression)

**Why it's good:**

* Fast to train
* Hard to overfit
* Great baseline model

### Random Forest Advantages

* Handles missing values
* Works well with categorical & numerical data
* Fast and easy to tune

### Random Forest — Real-World Applications

* **Social Media:** Spam and fake account detection
* **Healthcare:** Predicting disease diagnosis based on symptoms
* **Finance:** Credit scoring, loan approval systems
* **Environmental Science:** Predicting air quality, rainfall, or forest cover loss

---

## Boosting

**What is Boosting?**

* Models are trained sequentially
* Each new model focuses on the errors made by the previous ones
* Combines them into a strong model
* **Goal:** Reduce bias and improve prediction

### Boosting — Real-World Applications

* **Self-Driving Cars:** Object detection and decision-making
* **Email Providers:** Spam detection with high accuracy
* **Finance & Insurance:** Fraud detection (e.g., XGBoost for real-time alerts)
* **Gaming:** Predicting user churn or behavior

**Used in:**

* XGBoost
* LightGBM
* AdaBoost
* CatBoost

---

## Popular Boosting Algorithms

Boosting is an **ensemble learning technique**


| Algorithm    | Speed     | Overfitting Risk | Handles Categorical | Best For                |
| ------------ | --------- | ---------------- | ------------------- | ----------------------- |
| **AdaBoost** | Medium    | High (outliers)  | ❌ No                | Simple, small data      |
| **GBM**      | Slow      | High             | ❌ No                | Flexible loss functions |
| **XGBoost**  | Fast      | Medium           | ❌ No                | Accuracy, competitions  |
| **LightGBM** | Very Fast | Medium-High      | ❌ No                | Large datasets          |
| **CatBoost** | Medium    | Low              | ✅ Yes               | Mixed/categorical data  |

---