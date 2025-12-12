In this project I have done some data analysis project of my client 
And in data pipelines folder I have created some tutorical based on ColumnTransformer , Pipeline, Ordinal and OneHotEncoding, SimpleImputer method
In Carprice prediction folder I have developed one project in which I have done EDA, And used Custom Estimator Class to find R2_Scroe and predict the car price.
In app.py file I have design one UI Using Streamlit to predict the car price


Interview Notes: Log Transformation & Machine Learning
1. What Is Log Transformation?
A logarithmic transformation replaces a variable x with log(x) or log(1 + x). It is a non-linear scaling that
compresses large values, expands small ones, and reduces right skew.
2. Why Do We Use Log Transformation?
- Reduce Right Skewness - Makes data more symmetric
- Stabilize Variance - Reduces heteroscedasticity
- Linearize Relationships - Converts exponential patterns into linear
- Handle Wide Ranges - Prevents large numbers from dominating
- Improve Model Assumptions - Helps algorithms assuming linearity or normality
3. When Not to Use It
- Data contains negative values
- Data already normally distributed
- Categorical or binary data
- Left-skewed data (use square or exponential transforms instead)
4. On What Kind of Data It is Applied
- Continuous numeric (right-skewed): sales, income, price
- Negative/zero-heavy numeric: use log1p
- Categorical: do not apply
- Binary (0/1): not needed
5. Which Algorithms Benefit Most
- Linear Models (LinearRegression, Ridge, Lasso, LogisticRegression): Strong improvement
- Tree-Based Models (DecisionTree, RandomForest, XGBoost): Little or no effect
- Distance-Based Models (KNN, SVM, KMeans): Helpful for stability
- Neural Networks: Sometimes useful
- Statistical Models (OLS, ARIMA, GLM): Often required
6. Common Transform Functions in Python
import numpy as np
np.log1p(x) # log(1 + x)
np.expm1(y) # exp(y) - 1 (inverse)
from sklearn.preprocessing import FunctionTransformer
log_trf = FunctionTransformer(np.log1p)
X_log = log_trf.fit_transform(X)
7. Interview Quick Answers
Q1. Why apply log transform? -> Reduce skewness, linearize relationships, stabilize variance.
Q2. When not to use it? -> When data is normal or contains negatives.
Q3. Which models benefit most? -> Linear and distance-based.
Q4. Difference log(x) vs log1p(x)? -> log1p handles zeros.
Q5. Reverse transform? -> np.expm1(y_pred).
8. Bonus Tip - Log Transform in Regression
If target variable y is skewed (e.g., prices):
y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)
y_pred = np.expm1(model.predict(X_test))
This improves R^2 and stabilizes model behavior