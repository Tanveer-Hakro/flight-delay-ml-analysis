# Project Report: Flight Delay Prediction and Classification

## 1. Objective

This project explores a large-scale flight-delay dataset through exploratory analysis, feature engineering, regression, regularization, and linear classification techniques. The goal is to understand what can be learned from schedule, calendar, route-distance, and airline information when predicting departure delay and identifying flights delayed by more than 15 minutes.

## 2. Data and exploratory analysis

The executed notebook loads `Combined_Flights_2022.csv`, containing 4,078,318 flight records and 61 source columns. The analysis inspects summary statistics, missing values, departure-delay distributions, outliers, and correlations among numeric variables.

Feature engineering derives `day_of_week`, `month`, and `ScheduledDepHour` from the original date and scheduled-departure fields.

## 3. Regression

The regression target is `DepDelay` in minutes. The primary linear pipeline uses standardized numeric features, one-hot encoded airline information, and Ridge regression with `alpha` tuned by 3-fold cross-validation. The best tested value was `alpha = 0.1`.

The tuned Ridge model achieved **R² = 0.017** and **RMSE = 53.041 minutes**. Polynomial degree-2 expansion followed by OLS, Ridge, and Lasso produced nearly identical results, indicating that increasing polynomial complexity and applying coefficient shrinkage did not materially improve predictive performance with the current feature set.

## 4. Classification

A binary label marks a flight as delayed when `DepDelay > 15` minutes. The project compares Linear Discriminant Analysis, Logistic Regression, and Gaussian Naive Bayes using standardized numeric features and one-hot encoded airline information.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LDA | 0.795 | 0.502 | 0.018 | 0.035 |
| Logistic Regression | 0.795 | 0.498 | 0.010 | 0.020 |
| Naive Bayes | 0.687 | 0.313 | 0.442 | 0.367 |

The results show an important trade-off: LDA and Logistic Regression have higher overall accuracy, but detect very few delayed flights. Naive Bayes has lower accuracy yet substantially higher recall and F1 for the delayed class.

## 5. Main conclusion

The selected schedule and route features capture only a small fraction of the variation in departure delays. Important external drivers such as weather and air-traffic conditions are not represented, which limits regression performance. For classification, overall accuracy should be interpreted together with recall, F1, and the confusion matrix because delayed flights are the minority class in the test set.
