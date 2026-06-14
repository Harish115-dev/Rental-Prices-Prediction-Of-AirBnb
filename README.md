# Airbnb Price Prediction using Machine Learning

## Project Overview

This project predicts Airbnb listing prices using Machine Learning techniques. The dataset contains information about Airbnb listings such as location, room type, availability, and host information.

The objective is to build regression models capable of estimating listing prices and compare their performance using various evaluation metrics.

---

## Dataset Features

The following features were used:

* neighbourhood_group
* latitude
* longitude
* room_type
* minimum_nights
* calculated_host_listings_count
* availability_365

Target Variable:

* price

To reduce the effect of extreme price values, a logarithmic transformation was applied:

```python
data["price_log"] = np.log1p(data["price"])
```

---

## Data Preprocessing

### 1. Data Cleaning

Removed irrelevant columns:

* id
* host_id
* name
* host_name
* neighbourhood
* last_review
* reviews_per_month
* number_of_reviews

### 2. Missing Value Analysis

Checked dataset for missing values.

### 3. Outlier Analysis

Used boxplots and IQR analysis to investigate price outliers.

### 4. Log Transformation

Applied log transformation to the target variable:

```python
data["price_log"] = np.log1p(data["price"])
```

This helped reduce skewness and minimize the impact of extreme price values.

### 5. Encoding

Categorical features were converted into numerical format using One-Hot Encoding:

```python
pd.get_dummies()
```

### 6. Feature Scaling

StandardScaler was used before training Linear Regression.

---

## Models Used

### Linear Regression

A baseline regression model used to establish initial performance.

### Random Forest Regressor

An ensemble tree-based model capable of learning non-linear relationships.

### XGBoost Regressor

A gradient boosting algorithm that often performs well on structured tabular datasets.

---

## Evaluation Metrics

The following regression metrics were used:

### MAE (Mean Absolute Error)

Measures average prediction error.

### RMSE (Root Mean Squared Error)

Penalizes larger prediction errors more heavily.

### R² Score

Measures how much variation in the target variable is explained by the model.

---

## Results

| Model             | MAE   | RMSE   | R²    |
| ----------------- | ----- | ------ | ----- |
| Linear Regression | 58.94 | 164.10 | 0.147 |
| Random Forest     | 54.90 | 157.62 | 0.213 |
| XGBoost           | 52.85 | 156.30 | 0.226 |

### Best Model

XGBoost achieved the best performance among the evaluated models.

* Lowest MAE
* Lowest RMSE
* Highest R² Score

---

## Key Findings

* Log transformation significantly improved target distribution.
* One-Hot Encoding performed better than Label Encoding for categorical features.
* Tree-based models outperformed Linear Regression.
* XGBoost achieved the best overall predictive performance.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* XGBoost

---

## Future Improvements

Possible improvements include:

* Hyperparameter tuning using GridSearchCV
* Feature engineering
* Adding additional listing information
* Cross-validation
* Advanced ensemble techniques

---

## Author

Harish Rathwa
