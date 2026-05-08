import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("credit_data.csv")

print("\n================================================")
print("           CREDIT SCORING DATASET")
print("================================================\n")

print(data)


data = data.dropna()


le = LabelEncoder()

data['Payment_History'] = le.fit_transform(data['Payment_History'])
data['Creditworthy'] = le.fit_transform(data['Creditworthy'])


data['Debt_Income_Ratio'] = data['Debts'] / data['Income']


X = data.drop('Creditworthy', axis=1)
y = data['Creditworthy']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


def evaluate_model(model_name, model):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("\n================================================")
    print(f"         {model_name}")
    print("================================================")

    print(f"\nAccuracy Score  : {accuracy:.2f}")
    print(f"Precision Score : {precision:.2f}")
    print(f"Recall Score    : {recall:.2f}")
    print(f"F1 Score        : {f1:.2f}")
    print(f"ROC-AUC Score   : {roc:.2f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return accuracy


lr_model = LogisticRegression(max_iter=1000)

lr_accuracy = evaluate_model(
    "LOGISTIC REGRESSION MODEL",
    lr_model
)


dt_model = DecisionTreeClassifier(random_state=42)

dt_accuracy = evaluate_model(
    "DECISION TREE MODEL",
    dt_model
)


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_accuracy = evaluate_model(
    "RANDOM FOREST MODEL",
    rf_model
)


accuracies = {
    "Logistic Regression": lr_accuracy,
    "Decision Tree": dt_accuracy,
    "Random Forest": rf_accuracy
}

best_model = max(accuracies, key=accuracies.get)

print("\n================================================")
print("             BEST PERFORMING MODEL")
print("================================================\n")

print(f"{best_model} achieved the highest accuracy.")


sample_customer = pd.DataFrame({
    'Income': [55000],
    'Debts': [10000],
    'Payment_History': [2],
    'Credit_Utilization': [35],
    'Loan_Amount': [18000],
    'Employment_Years': [5],
    'Age': [30],
    'Debt_Income_Ratio': [10000 / 55000]
})

prediction = rf_model.predict(sample_customer)

print("\n================================================")
print("           SAMPLE CUSTOMER PREDICTION")
print("================================================\n")

if prediction[0] == 1:
    print("Customer is CREDITWORTHY")
else:
    print("Customer is NOT CREDITWORTHY")


print("\n================================================")
print("      CREDIT SCORING MODEL COMPLETED")
print("================================================\n")