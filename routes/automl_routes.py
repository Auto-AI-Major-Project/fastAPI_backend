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
