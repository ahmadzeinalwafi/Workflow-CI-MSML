name: Train and Deploy Model with MLflow Docker

on:
  push:
    branches: [master]
  workflow_dispatch:

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest

    env:
      MLFLOW_TRACKING_USERNAME: ahmadzeinalwafi
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run training script and save model
      run: python MLProject/modelling.py

    - name: Get latest MLflow run_id
      run: |
        RUN_ID=$(ls -td mlruns/0/*/ | head -n 1 | cut -d'/' -f3)
        echo "RUN_ID=$RUN_ID" >> $GITHUB_ENV
        echo "Latest run_id: $RUN_ID"

    - name: Build MLflow Docker image
      run: |
        mlflow models build-docker --model-uri runs:/$RUN_ID/model --name equehours/ml-model:latest

    - name: Log in to DockerHub
      uses: docker/login-action@v2
      with:
        username: equehours
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Push Docker image
      run: |
        docker push equehours/ml-model:latest
