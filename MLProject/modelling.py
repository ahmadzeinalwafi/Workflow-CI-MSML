import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_gamma_deviance,
    mean_pinball_loss,
    mean_poisson_deviance
)
from mlflow.models.signature import infer_signature

random_state = 42
train_path = "train_data.csv"
test_path = "test_data.csv"
artifact_data_path = "MLProject"
model_artifact_path = "model"

# Load pre-split train and test sets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop("strength", axis=1)
y_train = train_df["strength"]
X_test = test_df.drop("strength", axis=1)
y_test = test_df["strength"]

def train_and_log_model(run_name_suffix, tracking_uri=None):
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name=f"RandomForest_Baseline_{run_name_suffix}"):
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Log custom metrics
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("input_features", X_train.shape[1])
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("rmse_test", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("r2_test", r2_score(y_test, y_pred))
        mlflow.log_metric("mgd_test", mean_gamma_deviance(y_test, y_pred))
        mlflow.log_metric("mpl_test", mean_pinball_loss(y_test, y_pred))
        mlflow.log_metric("mpd_test", mean_poisson_deviance(y_test, y_pred))

        # Log model with input example and signature
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            model,
            artifact_path=model_artifact_path,
            input_example=input_example,
            signature=signature
        )

if __name__ == "__main__":
    train_and_log_model(
        # tracking_uri="sqlite:///mlruns.db",
        run_name_suffix="local"
    )

    # train_and_log_model(
    #     tracking_uri="https://dagshub.com/ahmadzeinalwafi/membangun-sistem-machine-learning.mlflow",
    #     run_name_suffix="dagshub"
    # )

