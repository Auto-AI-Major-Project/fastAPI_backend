from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from pycaret.classification import setup, compare_models, pull, finalize_model, save_model
import pandas as pd, numpy as np, tempfile, os, time, json
from datetime import datetime
from io import BytesIO
from database import get_db, AutoMLRun, ModelMetrics, User
from auth import get_current_user

router = APIRouter(prefix="/automl", tags=["AutoML"])

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Upload & Run AutoML ===
@router.post("")
async def automl_pipeline(file: UploadFile = File(...), target_col: str = Form(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(await file.read())
    tmp.close()

    df = pd.read_csv(tmp.name)
    setup(data=df, target=target_col, session_id=42, html=False, log_experiment=False)

    start_time = time.time()
    top_models = compare_models(sort="Accuracy", n_select=5)
    elapsed = round(time.time() - start_time, 2)
    results_df = pull()
    best_model = finalize_model(top_models[0])

    model_name = f"{best_model.__class__.__name__}_model"
    model_path = os.path.join(MODEL_DIR, model_name)
    save_model(best_model, model_path)

    recommendations = [
        {"name": row["Model"], "Accuracy": float(row["Accuracy"]), "AUC": float(row.get("AUC", 0)), "F1": float(row.get("F1", 0)), "TrainTime": f"{elapsed}s"}
        for _, row in results_df.iterrows()
    ]

    automl_run = AutoMLRun(
        user_id=current_user.id, filename=file.filename, target_column=target_col,
        top_model=best_model.__class__.__name__, model_path=model_path + ".pkl",
        processing_time=elapsed, recommendations=recommendations
    )
    db.add(automl_run)
    db.commit()
    db.refresh(automl_run)

    for _, row in results_df.iterrows():
        db.add(ModelMetrics(run_id=automl_run.id, model_name=row["Model"], accuracy=float(row["Accuracy"]),
                            auc=float(row.get("AUC", 0)), f1_score=float(row.get("F1", 0)), train_time=f"{elapsed}s"))
    db.commit()
    os.unlink(tmp.name)
    return {"status": "success", "run_id": automl_run.id, "recommendations": recommendations}


# === Get user’s runs ===
@router.get("/runs")
def get_all_runs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    runs = db.query(AutoMLRun).filter(AutoMLRun.user_id == current_user.id).all()
    return {"runs": runs}


# === Get single run ===
@router.get("/runs/{run_id}")
def get_run_details(run_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    run = db.query(AutoMLRun).filter(AutoMLRun.id == run_id, AutoMLRun.user_id == current_user.id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    metrics = db.query(ModelMetrics).filter(ModelMetrics.run_id == run_id).all()
    return {"run": run, "metrics": metrics}


# === Download model file ===
@router.get("/download-model/{run_id}")
def download_model(run_id: int, db: Session = Depends(get_db)):
    run = db.query(AutoMLRun).filter(AutoMLRun.id == run_id).first()
    if not run or not os.path.exists(run.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    filename = os.path.basename(run.model_path)
    return FileResponse(run.model_path, media_type="application/octet-stream", filename=filename)


# === Export results to Excel ===
@router.get("/export-results/{run_id}")
async def export_results(run_id: int, db: Session = Depends(get_db)):
    run = db.query(AutoMLRun).filter(AutoMLRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    metrics = db.query(ModelMetrics).filter(ModelMetrics.run_id == run_id).all()
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary = pd.DataFrame({"Property": ["Run ID", "Filename", "Target", "Best Model", "Time"], "Value": [run.id, run.filename, run.target_column, run.top_model, run.processing_time]})
        summary.to_excel(writer, sheet_name="Summary", index=False)
    output.seek(0)
    filename = f"AutoML_Run_{run_id}_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename={filename}"})



@router.post("/analyze-dataset")
async def analyze_dataset(
    file: UploadFile = File(...),
    target_col: str = Form(...)
):
    """
    Analyze dataset and return visualization data
    """
    # Save uploaded file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    contents = await file.read()
    tmp.write(contents)
    tmp.close()
    
    # Load dataset
    df = pd.read_csv(tmp.name)
    
    # Basic info
    dataset_info = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
        "target_column": target_col
    }
    
    # Column classifications
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Statistical summary for numeric columns
    numeric_summary = {}
    for col in numeric_cols:
        numeric_summary[col] = {
            "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
            "median": float(df[col].median()) if not df[col].isnull().all() else None,
            "std": float(df[col].std()) if not df[col].isnull().all() else None,
            "min": float(df[col].min()) if not df[col].isnull().all() else None,
            "max": float(df[col].max()) if not df[col].isnull().all() else None,
            "q25": float(df[col].quantile(0.25)) if not df[col].isnull().all() else None,
            "q75": float(df[col].quantile(0.75)) if not df[col].isnull().all() else None,
        }
    
    # Categorical summary
    categorical_summary = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts().head(10)
        categorical_summary[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in value_counts.items()}
        }
    
    # Target distribution
    target_distribution = {}
    if target_col in df.columns:
        if df[target_col].dtype in [np.int64, np.float64] and df[target_col].nunique() < 20:
            # Classification target
            target_counts = df[target_col].value_counts()
            target_distribution = {
                "type": "classification",
                "values": {str(k): int(v) for k, v in target_counts.items()}
            }
        elif df[target_col].dtype in [np.int64, np.float64]:
            # Regression target
            target_distribution = {
                "type": "regression",
                "stats": {
                    "mean": float(df[target_col].mean()),
                    "median": float(df[target_col].median()),
                    "std": float(df[target_col].std()),
                    "min": float(df[target_col].min()),
                    "max": float(df[target_col].max())
                },
                "histogram": df[target_col].value_counts(bins=20).to_dict()
            }
        else:
            # Categorical target
            target_counts = df[target_col].value_counts()
            target_distribution = {
                "type": "categorical",
                "values": {str(k): int(v) for k, v in target_counts.items()}
            }
    
    # Correlation matrix for numeric columns
    correlation_matrix = {}
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        correlation_matrix = {
            "columns": numeric_cols,
            "data": corr.values.tolist()
        }
    
    # Feature distributions (histograms for numeric features)
    feature_distributions = {}
    for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
        hist, bins = np.histogram(df[col].dropna(), bins=20)
        feature_distributions[col] = {
            "counts": hist.tolist(),
            "bins": bins.tolist()
        }
    
    # Categorical feature distributions
    categorical_distributions = {}
    for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
        value_counts = df[col].value_counts().head(10)
        categorical_distributions[col] = {
            "labels": value_counts.index.tolist(),
            "values": value_counts.values.tolist()
        }
    
    # Box plot data for numeric features
    box_plot_data = {}
    for col in numeric_cols[:10]:
        box_plot_data[col] = {
            "min": float(df[col].min()),
            "q1": float(df[col].quantile(0.25)),
            "median": float(df[col].median()),
            "q3": float(df[col].quantile(0.75)),
            "max": float(df[col].max()),
            "outliers": df[col][(df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) | 
                                 (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))].tolist()[:50]
        }
    
    # Scatter plot data (first 2 numeric features vs target if numeric)
    scatter_data = []
    if len(numeric_cols) >= 2 and target_col in numeric_cols:
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size)
        for col in numeric_cols[:2]:
            if col != target_col:
                scatter_data.append({
                    "feature": col,
                    "x": sample_df[col].tolist(),
                    "y": sample_df[target_col].tolist()
                })
    
    # Clean up
    os.unlink(tmp.name)
    
    return {
        "dataset_info": dataset_info,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "target_distribution": target_distribution,
        "correlation_matrix": correlation_matrix,
        "feature_distributions": feature_distributions,
        "categorical_distributions": categorical_distributions,
        "box_plot_data": box_plot_data,
        "scatter_data": scatter_data
    }

@router.get("/analyze-run/{run_id}")
async def analyze_run_dataset(run_id: int, db: Session = Depends(get_db)):
    """
    Get visualization data for a previously run AutoML job
    """
    run = db.query(AutoMLRun).filter(AutoMLRun.id == run_id).first()
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Return stored analysis data or prompt to re-upload dataset
    return {
        "message": "Please re-upload the dataset for visualization analysis",
        "run_info": {
            "filename": run.filename,
            "target_column": run.target_column,
            "top_model": run.top_model
        }
    }