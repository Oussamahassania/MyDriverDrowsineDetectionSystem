import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the trained Logistic Regression model
model = joblib.load('logistic_regression_awake_drowsy_classifier.joblib')

# Assuming you have a test dataset with features X_test and true labels y_true
# Replace these with your actual test data
# For example:
X_test = # Your test features data
y_true = # Your true labels

# Predict the labels using the loaded model
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, model.predict_proba(X_test)[:, 1])  # Using probabilities for ROC-AUC

# Displaying the results
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")
