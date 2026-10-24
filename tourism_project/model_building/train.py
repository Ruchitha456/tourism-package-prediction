
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

from xgboost import XGBClassifier
import joblib

# Load prepared training and testing datasets
X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")
y_train = pd.read_csv("ytrain.csv").squeeze("columns")
y_test = pd.read_csv("ytest.csv").squeeze("columns")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Identify numerical and categorical features
categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

numerical_features = X_train.select_dtypes(include=["number"]).columns.tolist()


# Preprocess both feature types without leaking test data into training
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numerical_features),
        ("categorical",OneHotEncoder(handle_unknown="ignore"),categorical_features)
    ]
)
# Build a preprocessing and classification pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("classifier",XGBClassifier(objective = "binary:logistic", eval_metric = "logloss", random_state=42, n_jobs=1))
    ]
)
# Define a small hyperparameter grid for cross-validation
param_grid = {
    "classifier__n_estimators": [100, 150],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1]
}
# Configure MLflow experiment tracking
mlflow.set_experiment("Tourism_Package_Prediction")
with mlflow.start_run(run_name="XGBoost_GridSearch"):
     search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
     )
     search.fit(X_train,y_train)
     best_model = search.best_estimator_
     predictions = best_model.predict(X_test)
     # Calculate test-set evaluation metrics
     metrics = {
               "accuracy": accuracy_score(y_test,predictions),
               "precision": precision_score(y_test,predictions,zero_division=0),
               "recall": recall_score(y_test,predictions,zero_division=0),
               "f1_score": f1_score(y_test,predictions,zero_division=0)
               } 

     # Log the best hyperparameters and evaluation metrics
     mlflow.log_params(search.best_params_)
     mlflow.log_param("cross_validation_folds",3)
     mlflow.log_param("scoring_metric","f1")
     mlflow.log_metrics(metrics)

     print("\nBest hyperparameters:")
     print(search.best_params_)

     print("\nTest-set metrics:")
     for metric_name, metric_value in metrics.items():
        print(f"{metric_name}:{metric_value:.4f}")

     print("\nClassification report:")
     print(classification_report(y_test,predictions,zero_division=0))

     # Save the complete fitted pipeline for deployment
     output_dir = "tourism_project/deployment"
     os.makedirs(output_dir,exist_ok=True)

     model_path = os.path.join(output_dir,"tourism_model.joblib")
     joblib.dump(best_model,model_path)
     
     # Log the trained model as an MLflow artifact
     mlflow.log_artifact(model_path)
     print("Trained model logged as an MLflow artifact.")
     
     print("\nTrained model saved to:", model_path)
     print("MLflow run ID:", mlflow.active_run().info.run_id)
