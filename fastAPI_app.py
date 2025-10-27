
from http.client import HTTPException
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse , JSONResponse
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from pycaret.classification import setup, compare_models, pull, finalize_model, save_model
import pandas as pd
import tempfile
import time
import os
from database import get_db, create_tables, AutoMLRun, ModelMetrics
from fastapi.responses import StreamingResponse
from io import BytesIO
from datetime import datetime
import numpy as np
from typing import List, Dict, Any
import json


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


@app.get("/automl/download-model/{run_id}")
async def download_model(run_id: int, db: Session = Depends(get_db)):
    """
    Download the trained model file for a specific run
    """
    run = db.query(AutoMLRun).filter(AutoMLRun.id == run_id).first()
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    model_path = run.model_path
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found on server")
    
    # Extract filename from path
    filename = os.path.basename(model_path)
    
    return FileResponse(
        path=model_path,
        media_type='application/octet-stream',
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/automl/export-results/{run_id}")
async def export_results(run_id: int, db: Session = Depends(get_db)):
    """
    Export AutoML results to Excel file with multiple sheets
    """
    run = db.query(AutoMLRun).filter(AutoMLRun.id == run_id).first()
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    metrics = db.query(ModelMetrics).filter(ModelMetrics.run_id == run_id).all()
    
    # Create Excel file in memory
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: Run Summary
        summary_data = {
            'Property': [
                'Run ID',
                'Filename',
                'Target Column',
                'Best Model',
                'Processing Time',
                'Created At',
                'Status',
                'Model Path'
            ],
            'Value': [
                run.id,
                run.filename,
                run.target_column,
                run.top_model,
                f"{run.processing_time}s",
                str(run.created_at),
                'Success',
                run.model_path
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Run Summary', index=False)
        
        # Sheet 2: Model Comparison
        if run.recommendations:
            models_data = []
            for idx, rec in enumerate(run.recommendations):
                models_data.append({
                    'Rank': idx + 1,
                    'Model Name': rec.get('name', 'N/A'),
                    'Accuracy': f"{rec.get('Accuracy', 0) * 100:.2f}%",
                    'AUC': f"{rec.get('AUC', 0) * 100:.2f}%" if rec.get('AUC') else 'N/A',
                    'F1 Score': f"{rec.get('F1', 0) * 100:.2f}%" if rec.get('F1') else 'N/A',
                    'Train Time': rec.get('TrainTime', 'N/A'),
                    'Best Model': '✓' if idx == 0 else ''
                })
            models_df = pd.DataFrame(models_data)
            models_df.to_excel(writer, sheet_name='Model Comparison', index=False)
        
        # Sheet 3: Detailed Metrics (from database)
        if metrics:
            detailed_data = []
            for metric in metrics:
                detailed_data.append({
                    'Model Name': metric.model_name,
                    'Accuracy': f"{metric.accuracy * 100:.2f}%" if metric.accuracy else 'N/A',
                    'AUC': f"{metric.auc * 100:.2f}%" if metric.auc else 'N/A',
                    'F1 Score': f"{metric.f1_score * 100:.2f}%" if metric.f1_score else 'N/A',
                    'Train Time': metric.train_time
                })
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        
        # Sheet 4: Full Raw Data
        raw_data = {
            'recommendations': str(run.recommendations),
            'processing_time': run.processing_time,
            'created_at': str(run.created_at)
        }
        raw_df = pd.DataFrame([raw_data])
        raw_df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        # Format the workbook
        workbook = writer.book
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#1A6B8E',
            'font_color': 'white',
            'border': 1
        })
        
        # Apply formatting to all sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:Z', 20)  # Set column width
            
            # Format header row
            for col_num, value in enumerate(summary_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
    
    output.seek(0)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"AutoML_Results_Run{run_id}_{timestamp}.xlsx"
    
    return StreamingResponse(
        output,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/automl/analyze-dataset")
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

@app.get("/automl/analyze-run/{run_id}")
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

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FastAPI AutoML Backend"}