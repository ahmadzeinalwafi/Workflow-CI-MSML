# modelling.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_gamma_deviance, mean_pinball_loss, mean_poisson_deviance
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("https://dagshub.com/ahmadzeinalwafi/membangun-sistem-machine-learning.mlflow")
mlflow.sklearn.autolog()
random_state = 42

if __name__ == "__main__":
    df = pd.read_csv("MLProject/dataset_clean.csv")

    X = df.drop("strength", axis=1)
    y = df["strength"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mgd = mean_gamma_deviance(y_test, y_pred)
        mpl = mean_pinball_loss(y_test, y_pred)
        mpd = mean_poisson_deviance(y_test, y_pred)

        mlflow.autolog()

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
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
        mlflow.log_artifact("MLProject/dataset_clean.csv", artifact_path="data")
