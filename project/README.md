# Compiling instruction

1. Setting up the environment via:
   ```bash 
    cd ML-DL-application/project
    #create an env via conda or default python (for me I use conda)

    conda create -n ml
    conda activate ml

    pip install -r requirements.txt
   ```
2. Run experiments via:
    ```bash 
    python3 Main_ML.py
    ```
3. To track down the training process as well as view the results, run: 
   ```bash 
   mlflow ui --backend-store-uri "file:///media/gamedisk/ML&DL/project/mlruns (Replace your absolute path here)" --port 10001
    ```


