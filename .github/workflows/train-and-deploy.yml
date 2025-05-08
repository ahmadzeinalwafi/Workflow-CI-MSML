name: Train and Deploy Model with MLflow Docker

on:
  push:
    branches: [master]
  workflow_dispatch:
    
env:
  MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING: false
  MLFLOW_AUTOLOG_INPUT_DATASETS: false
  
jobs:
  train-and-deploy:
    runs-on: ubuntu-latest

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
      run: mlflow run MLProject --env-manager=local 

    - name: Get latest MLflow run_id
      run: |
        RUN_ID=$(ls -td mlruns/0/*/ | head -n 1 | cut -d'/' -f3)
        echo "RUN_ID=$RUN_ID" >> $GITHUB_ENV
        echo "Latest run_id: $RUN_ID"
    
    - name: Save the artifact to GitHub
      uses: actions/upload-artifact@v4
      with:
        name: artifact
        path: ./mlruns

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
