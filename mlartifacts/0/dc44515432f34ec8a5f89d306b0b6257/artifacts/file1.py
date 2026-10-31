import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 

mlflow.set_tracking_uri("http://localhost:5000")

wine = load_wine()
x = wine.data
y = wine.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

max_depth = 10
n_estimators = 10

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"Accuracy: {accuracy}")

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.figure(figsize=(6, 6))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    mlflow.log_artifact(__file__)

    # mlflow.sklearn.log_model(rf, "random_forest_model")