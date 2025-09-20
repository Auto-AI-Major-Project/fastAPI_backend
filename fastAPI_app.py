# # fastapi_app.py
# from fastapi import FastAPI, UploadFile, File, Form
# from pycaret.classification import setup, compare_models, pull
# import pandas as pd
# import tempfile
# import time

# app = FastAPI()

# @app.post("/recommendations")
# async def get_recommendations(file: UploadFile = File(...), target_col: str = Form(...)):
#     """
#     Upload a CSV + target column name -> run AutoML -> return model recommendations
#     """

#     # Save uploaded file temporarily
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
#     contents = await file.read()
#     tmp.write(contents)
#     tmp.close()

#     # Load dataset
#     df = pd.read_csv(tmp.name)

#     # Setup PyCaret environment
#     setup(data=df, target=target_col, session_id=42, html=False, log_experiment=False)

#     # Compare models (will train multiple algorithms)
#     start = time.time()
#     best_model = compare_models(sort="Accuracy", n_select=5)  # get top 5 models
#     elapsed = round(time.time() - start, 2)

#     # Extract results table
#     results = pull()  # returns Pandas dataframe of models and metrics

#     # Prepare response in {name, score, time} format
#     recommendations = []
#     for _, row in results.iterrows():
#         recommendations.append({
#             "name": row["Model"],
#             "score": float(row["Accuracy"]),   # pick Accuracy for now
#             "time": f"{elapsed:.2f}s"
#         })

#     return recommendations







# fastapi_app.py
from fastapi import FastAPI, UploadFile, File, Form
from pycaret.classification import setup, compare_models, pull, finalize_model, save_model
import pandas as pd
import tempfile
import time
import os

app = FastAPI()

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.post("/automl")
async def automl_pipeline(file: UploadFile = File(...), target_col: str = Form(...)):
    """
    Full AutoML pipeline:
    1. Load CSV
    2. Data cleaning & preprocessing
    3. AutoML model comparison
    4. Final model training
    5. Return top model recommendations + metrics
    """

    # Step 1: Save uploaded file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    contents = await file.read()
    tmp.write(contents)
    tmp.close()

    # Step 2: Load dataset
    df = pd.read_csv(tmp.name)

    # Step 3: Setup PyCaret environment (includes preprocessing)
    clf_setup = setup(
        data=df,
        target=target_col,
        session_id=42,
        html=False,
        log_experiment=False
    )

    # Step 4: Compare models
    start_time = time.time()
    top_models = compare_models(sort="Accuracy", n_select=5)
    elapsed = round(time.time() - start_time, 2)

    # Step 5: Pull metrics
    results_df = pull()  # dataframe with all model metrics

    # Step 6: Finalize best model (highest accuracy)
    best_model = top_models[0]
    finalized_model = finalize_model(best_model)

    # Optional: save the finalized model
    model_name = f"{best_model.__class__.__name__}_model"
    model_path = os.path.join(MODEL_DIR, model_name)
    save_model(finalized_model, model_path)

    # Step 7: Prepare API response
    recommendations = []
    for _, row in results_df.iterrows():
        recommendations.append({
            "name": row["Model"],
            "Accuracy": float(row["Accuracy"]),
            "AUC": float(row.get("AUC", 0.0)),
            "F1": float(row.get("F1", 0.0)),
            "TrainTime": f"{elapsed:.2f}s"
        })

    response = {
        "top_model": best_model.__class__.__name__,
        "model_path": model_path + ".pkl",
        "recommendations": recommendations
    }

    return response
