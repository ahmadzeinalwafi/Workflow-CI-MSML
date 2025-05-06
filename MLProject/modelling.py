import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_gamma_deviance,
    mean_pinball_loss,
    mean_poisson_deviance
)
from mlflow.models.signature import infer_signature

random_state = 42
dataset_path = "MLProject/dataset_clean.csv"
artifact_data_path = "MLProject/dataset_clean.csv"
model_artifact_path = "model"

df = pd.read_csv(dataset_path)
X = df.drop("strength", axis=1)
y = df["strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

def train_and_log_model(run_name_suffix, tracking_uri = None):
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name=f"RandomForest_Baseline_{run_name_suffix}"):
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Custom metrics
        rmse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mgd = mean_gamma_deviance(y_test, y_pred)
        mpl = mean_pinball_loss(y_test, y_pred)
        mpd = mean_poisson_deviance(y_test, y_pred)

        # Custom logs
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("input_features", X.shape[1])
        mlflow.log_param("split_ratio", "80/20")
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("rmse_test", rmse)
        mlflow.log_metric("r2_test", r2)
        mlflow.log_metric("mgd_test", mgd)
        mlflow.log_metric("mpl_test", mpl)
        mlflow.log_metric("mpd_test", mpd)

        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            model,
            artifact_path=model_artifact_path,
            input_example=input_example,
            signature=signature
        )

        mlflow.log_artifact(artifact_data_path, artifact_path="data")

if __name__ == "__main__":
    # Log locally to ./mlruns
    train_and_log_model(
        run_name_suffix="local"
    )

    # Save artifacts to repository
    train_and_log_model(
        tracking_uri="https://dagshub.com/ahmadzeinalwafi/membangun-sistem-machine-learning.mlflow",
        run_name_suffix="dagshub"
    )

