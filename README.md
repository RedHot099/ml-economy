# Credit Card Fraud Detection Results


This notebook presents the results of several machine learning models applied to
the credit card fraud detection dataset. The dataset was split into training
and testing sets with an 80/20 ratio, stratifying to preserve the class
distribution. SMOTE (from `imblearn.over_sampling`) was applied to the training set to address class imbalance.


## Model Performance


The following table summarizes the performance of each model:


| Model               | Confusion Matrix                | Recall | % of Detected Frauds |
| ------------------- | --------------------------------- | ------ | -------------------- |
| kNN                 | [[56753  111]  [ 12   86]]          | 0.88 | 87.76%               |
| GaussianNB          | [[55418 1446]  [ 12   86]]          | 0.88 | 87.76%               |
| Logistic Regression | [[32000 24864]  [  1   97]]          | 0.99 | 98.99%               |
| Autoencoder         | [[54096 2768]  [ 17   81]]          | 0.83 | 82.65%               |


**Note:** The performance of the Autoencoder is heavily dependent on the chosen threshold
for anomaly detection. This value will require further adjustments to be more accurate.


## Summary of Results


The Logistic Regression model (`sklearn.linear_model`) achieved the highest recall (99%), capturing nearly all fraudulent transactions. This superior performance is attributed to adjusting class weights in the imbalanced dataset, using `class_weight.compute_class_weight` (from `sklearn.utils`) with the 'balanced' setting to compute the weights and applying these with `class_weight` parameter in `LogisticRegression`.


However, this comes at the cost of very low precision, resulting in a high number of false alarms. If the business can tolerate a high false alarm rate to maximize fraud detection, Logistic Regression may be the preferred model. 
