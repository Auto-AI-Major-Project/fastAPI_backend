# from fastapi import FastAPI, UploadFile, File, Form
# from pycaret.classification import setup, compare_models, pull, finalize_model, save_model
# import pandas as pd
# import tempfile
# import time
# import os

# app = FastAPI()

# MODEL_DIR = "trained_models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# @app.post("/automl")
# async def automl_pipeline(file: UploadFile = File(...), target_col: str = Form(...)):
#     """
#     Full AutoML pipeline:
#     1. Load CSV
#     2. Data cleaning & preprocessing
#     3. AutoML model comparison
#     4. Final model training
#     5. Return top model recommendations + metrics
#     """

#     # Step 1: Save uploaded file
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
#     contents = await file.read()
#     tmp.write(contents)
#     tmp.close()

#     # Step 2: Load dataset
#     df = pd.read_csv(tmp.name)

#     # Step 3: Setup PyCaret environment (includes preprocessing)
#     clf_setup = setup(
#         data=df,
#         target=target_col,
#         session_id=42,
#         html=False,
#         log_experiment=False
#     )

#     # Step 4: Compare models
#     start_time = time.time()
#     top_models = compare_models(sort="Accuracy", n_select=5)
#     elapsed = round(time.time() - start_time, 2)

#     # Step 5: Pull metrics
#     results_df = pull()  # dataframe with all model metrics

#     # Step 6: Finalize best model (highest accuracy)
#     best_model = top_models[0]
#     finalized_model = finalize_model(best_model)

#     # Optional: save the finalized model
#     model_name = f"{best_model.__class__.__name__}_model"
#     model_path = os.path.join(MODEL_DIR, model_name)
#     save_model(finalized_model, model_path)

#     # Step 7: Prepare API response
#     recommendations = []
#     for _, row in results_df.iterrows():
#         recommendations.append({
#             "name": row["Model"],
#             "Accuracy": float(row["Accuracy"]),
#             "AUC": float(row.get("AUC", 0.0)),
#             "F1": float(row.get("F1", 0.0)),
#             "TrainTime": f"{elapsed:.2f}s"
#         })

#     response = {
#         "top_model": best_model.__class__.__name__,
#         "model_path": model_path + ".pkl",
#         "recommendations": recommendations
#     }

#     return response





from fastapi import FastAPI, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from pycaret.classification import setup, compare_models, pull, finalize_model, save_model
import pandas as pd
import tempfile
import time
import os
from database import get_db, create_tables, AutoMLRun, ModelMetrics


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React app URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Create database tables on startup
@app.on_event("startup")
def startup_event():
    create_tables()

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.post("/automl")
async def automl_pipeline(
    file: UploadFile = File(...), 
    target_col: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Full AutoML pipeline:
    1. Load CSV
    2. Data cleaning & preprocessing
    3. AutoML model comparison
    4. Final model training
    5. Save results to database
    6. Return top model recommendations + metrics
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

    # Step 7: Prepare API response data
    recommendations = []
    for _, row in results_df.iterrows():
        recommendations.append({
            "name": row["Model"],
            "Accuracy": float(row["Accuracy"]),
            "AUC": float(row.get("AUC", 0.0)),
            "F1": float(row.get("F1", 0.0)),
            "TrainTime": f"{elapsed:.2f}s"
        })

    # Step 8: Save to database
    try:
        # Create main AutoML run record
        automl_run = AutoMLRun(
            filename=file.filename,
            target_column=target_col,
            top_model=best_model.__class__.__name__,
            model_path=model_path + ".pkl",
            processing_time=elapsed,
            recommendations=recommendations
        )
        
        db.add(automl_run)
        db.commit()
        db.refresh(automl_run)
        
        # Save individual model metrics
        for _, row in results_df.iterrows():
            model_metric = ModelMetrics(
                run_id=automl_run.id,
                model_name=row["Model"],
                accuracy=float(row["Accuracy"]),
                auc=float(row.get("AUC", 0.0)) if row.get("AUC") is not None else None,
                f1_score=float(row.get("F1", 0.0)) if row.get("F1") is not None else None,
                train_time=f"{elapsed:.2f}s"
            )
            db.add(model_metric)
        
        db.commit()
        
        # Clean up temporary file
        os.unlink(tmp.name)
        
        # Step 9: Prepare API response
        response = {
            "run_id": automl_run.id,
            "top_model": best_model.__class__.__name__,
            "model_path": model_path + ".pkl",
            "processing_time": f"{elapsed:.2f}s",
            "recommendations": recommendations,
            "status": "success",
            "message": "AutoML pipeline completed and saved to database"
        }

        return response
        
    except Exception as db_error:
        # Rollback in case of database error
        db.rollback()
        # Clean up temporary file
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        
        # Still return the results even if database save fails
        response = {
            "top_model": best_model.__class__.__name__,
            "model_path": model_path + ".pkl",
            "processing_time": f"{elapsed:.2f}s",
            "recommendations": recommendations,
            "status": "partial_success",
            "message": f"AutoML completed but database save failed: {str(db_error)}"
        }
        
        return response

@app.get("/automl/runs")
def get_all_runs(db: Session = Depends(get_db)):
    """Get all AutoML runs from database"""
    runs = db.query(AutoMLRun).all()
    return {"runs": runs}

@app.get("/automl/runs/{run_id}")
def get_run_details(run_id: int, db: Session = Depends(get_db)):
    """Get specific AutoML run details with metrics"""
    run = db.query(AutoMLRun).filter(AutoMLRun.id == run_id).first()
    if not run:
        return {"error": "Run not found"}
    
    metrics = db.query(ModelMetrics).filter(ModelMetrics.run_id == run_id).all()
    
    return {
        "run": run,
        "metrics": metrics
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FastAPI AutoML Backend"}