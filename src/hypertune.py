from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

import os
from dotenv import load_dotenv
load_dotenv()

repo_owner = os.getenv("REPO_OWNER")
repo_name = os.getenv("REPO_NAME")
track_uri = os.getenv("TRACKING_URI")

import dagshub
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

mlflow.set_tracking_uri(track_uri)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = pd.DataFrame(data.target, columns=['target'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

rf = RandomForestClassifier(random_state = 42)
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [None, 5, 7, 9],}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

mlflow.set_experiment("RF_Breast_Cancer")
with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_param("best_params", best_params)
    mlflow.log_metric("best_accuracy", best_score)

    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")

    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(
                            sk_model=grid_search.best_estimator_,
                            artifact_path="random_forest",
                            skops_trusted_types=["sklearn.tree._tree.Tree"])
    mlflow.set_tags({"Author": "Canxy", "Project": "Breast Cancer Classifier"})

    print(best_params)
    print(best_score)