# data_cleaning_routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from io import BytesIO, StringIO
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from database import get_db, User, DataCleaningSession
from auth import get_current_user
from utils import data_utils, data_cleaner

router = APIRouter(prefix="/api/data-cleaning", tags=["Data Cleaning & EDA"])

# Available manual cleaning operations
MANUAL_CLEANING_OPERATIONS = [
    "remove_duplicates",
    "remove_missing_rows",
    "impute_median",
    "impute_mode",
    "one_hot_encode"
]

# Helper function to get dataframe from store with user validation
def get_df_from_user_store(filename: str, user_id: int, db: Session) -> pd.DataFrame:
    """Retrieves dataframe and validates user ownership"""
    # Check if session exists in database
    session = db.query(DataCleaningSession).filter(
        DataCleaningSession.filename == filename,
        DataCleaningSession.user_id == user_id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=404, 
            detail=f"Dataset '{filename}' not found or you don't have access to it."
        )
    
    # Get from in-memory store
    df = data_utils.global_data_store.get(f"{user_id}_{filename}")
    if df is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Dataset '{filename}' not found in memory. Please re-upload."
        )
    return df


@router.post("/upload-data")
async def upload_data_and_inspect(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    **Step 1: Upload and Inspect.**
    Uploads file, stores it in memory, and returns inspection data with cleaning options.
    """
    file_name = file.filename
    user_id = current_user.id
    
    try:
        content = await file.read()
        df = data_utils.load_data(content, file_name)
        
        # Store with user-specific key
        user_file_key = f"{user_id}_{file_name}"
        data_utils.global_data_store[user_file_key] = df
        
        # Get inspection data
        inspection_data = data_utils.get_dataset_inspection(df, file_name)
        
        # Save session to database
        existing_session = db.query(DataCleaningSession).filter(
            DataCleaningSession.filename == file_name,
            DataCleaningSession.user_id == user_id
        ).first()
        
        if existing_session:
            existing_session.upload_date = datetime.utcnow()
            existing_session.original_shape = str(df.shape)
            existing_session.status = "uploaded"
        else:
            new_session = DataCleaningSession(
                user_id=user_id,
                filename=file_name,
                original_shape=str(df.shape),
                status="uploaded"
            )
            db.add(new_session)
        
        db.commit()
        
        return JSONResponse(content={
            "file_name": file_name,
            "status": "Success - Data Inspected",
            "inspection": inspection_data,
            "cleaning_options": {
                "manual": {
                    "endpoint": f"/api/data-cleaning/clean-manual/{file_name}",
                    "available_operations": MANUAL_CLEANING_OPERATIONS,
                    "description": "Select specific actions by sending a list of operations."
                },
                "automatic": {
                    "endpoint": f"/api/data-cleaning/clean-automatic/{file_name}",
                    "description": "Full intelligent cleaning, preprocessing, and EDA reporting."
                }
            }
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File upload failed: {str(e)}")


@router.post("/clean-manual/{file_name}")
async def clean_manual(
    file_name: str,
    operations: List[str] = Query(..., description=f"Operations: {MANUAL_CLEANING_OPERATIONS}"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    **Manual Cleaning:** Applies user-specified operations.
    """
    user_id = current_user.id
    df = get_df_from_user_store(file_name, user_id, db)
    
    try:
        df_cleaned, summary = data_cleaner.apply_manual_cleaning(df, operations)
        
        # Store cleaned data
        cleaned_key = f"{user_id}_{file_name}_MANUAL_CLEANED"
        data_utils.global_data_store[cleaned_key] = df_cleaned
        
        # Update database session
        session = db.query(DataCleaningSession).filter(
            DataCleaningSession.filename == file_name,
            DataCleaningSession.user_id == user_id
        ).first()
        
        if session:
            session.cleaned_shape = str(df_cleaned.shape)
            session.cleaning_method = "manual"
            session.operations_applied = str(operations)
            session.status = "cleaned"
            session.cleaned_date = datetime.utcnow()
            db.commit()
        
        return JSONResponse(content={
            "status": "Manual Cleaning Complete",
            "operations_summary": summary,
            "original_shape": list(df.shape),
            "final_shape": list(df_cleaned.shape),
            "cleaned_filename": f"{file_name}_MANUAL_CLEANED",
            "next_steps": {
                "download_report": f"/api/data-cleaning/download-report/{file_name}_MANUAL_CLEANED",
                "download_data": f"/api/data-cleaning/download-data/{file_name}_MANUAL_CLEANED"
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual cleaning failed: {str(e)}")


@router.post("/clean-automatic/{file_name}")
async def clean_automatic(
    file_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    **Automated Cleaning:** Executes intelligent cleaning, detailed EDA, and prepares data.
    """
    user_id = current_user.id
    df = get_df_from_user_store(file_name, user_id, db)
    
    try:
        inspection_data = data_utils.get_dataset_inspection(df, file_name)
        
        # Execute the full pipeline
        df_cleaned, status, report_details = data_cleaner.automated_clean_and_report_logic(
            df, file_name, inspection_data
        )
        
        # Store cleaned data
        cleaned_key = f"{user_id}_{file_name}_AUTO_CLEANED"
        data_utils.global_data_store[cleaned_key] = df_cleaned
        
        # Update database session
        session = db.query(DataCleaningSession).filter(
            DataCleaningSession.filename == file_name,
            DataCleaningSession.user_id == user_id
        ).first()
        
        if session:
            session.cleaned_shape = str(df_cleaned.shape)
            session.cleaning_method = "automatic"
            session.operations_applied = str(report_details.get('cleaning_summary'))
            session.status = "cleaned"
            session.cleaned_date = datetime.utcnow()
            db.commit()
        
        return JSONResponse(content={
            "status": status,
            "message": "Automated cleaning and EDA complete.",
            "original_shape": list(df.shape),
            "final_shape": list(df_cleaned.shape),
            "cleaning_summary": report_details.get('cleaning_summary'),
            "key_insights": report_details.get('key_insight'),
            "cleaned_filename": f"{file_name}_AUTO_CLEANED",
            "next_steps": {
                "download_report": f"/api/data-cleaning/download-report/{file_name}_AUTO_CLEANED",
                "download_data": f"/api/data-cleaning/download-data/{file_name}_AUTO_CLEANED"
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Automated cleaning failed: {str(e)}")


# @router.get("/download-report/{file_name}")
# async def download_report(
#     file_name: str,
#     format: str = Query("docx", description="Report format: 'docx' or 'json'"),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Downloads the detailed cleaning/EDA report.
#     """
#     user_id = current_user.id
#     df = get_df_from_user_store(file_name, user_id, db)
    
#     try:
#         inspection_data = data_utils.get_dataset_inspection(df, file_name)
        
#         # Re-run logic to generate report
#         df_temp, status, report_details = data_cleaner.automated_clean_and_report_logic(
#             df, file_name, inspection_data
#         )
        
#         if status != "Success":
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Report generation failed: {report_details.get('key_insight', 'Unknown error')}"
#             )
        
#         if format.lower() == 'json':
#             return JSONResponse(content={"report_details": report_details})
        
#         if format.lower() == 'docx':
#             file_stream = data_cleaner.generate_docx_report(report_details, file_name)
            
#             return StreamingResponse(
#                 file_stream,
#                 media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
#                 headers={"Content-Disposition": f"attachment; filename={file_name}_REPORT.docx"}
#             )
        
#         raise HTTPException(status_code=400, detail="Unsupported format. Use 'docx' or 'json'.")
    
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")


# @router.get("/download-data/{file_name}")
# async def download_dataset(
#     file_name: str,
#     format: str = Query("csv", description="Format: 'csv' or 'excel'"),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """Downloads the cleaned dataset."""
#     user_id = current_user.id
#     df = get_df_from_user_store(file_name, user_id, db)
    
#     try:
#         if format.lower() == 'csv':
#             stream = StringIO()
#             df.to_csv(stream, index=False)
#             response_content = stream.getvalue().encode('utf-8')
#             media_type = "text/csv"
#             file_ext = "csv"
#         elif format.lower() == 'excel':
#             stream = BytesIO()
#             df.to_excel(stream, index=False, engine='openpyxl')
#             response_content = stream.getvalue()
#             media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             file_ext = "xlsx"
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported format.")
        
#         return StreamingResponse(
#             BytesIO(response_content),
#             media_type=media_type,
#             headers={"Content-Disposition": f"attachment; filename={file_name}_CLEANED.{file_ext}"}
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.get("/download-report/{file_name}")
async def download_report(
    file_name: str,
    format: str = Query("pdf", description="Report format: 'pdf' or 'json'"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Downloads the detailed cleaning/EDA report in PDF format.
    """
    user_id = current_user.id
    
    # Handle the cleaned filename - remove suffix to get original
    original_filename = file_name.replace('_AUTO_CLEANED', '').replace('_MANUAL_CLEANED', '')
    
    try:
        # First try to get the cleaned version
        cleaned_key = f"{user_id}_{file_name}"
        df = data_utils.global_data_store.get(cleaned_key)
        
        # If not found, try the original file
        if df is None:
            original_key = f"{user_id}_{original_filename}"
            df = data_utils.global_data_store.get(original_key)
        
        # If still not found, check database and raise error
        if df is None:
            session = db.query(DataCleaningSession).filter(
                DataCleaningSession.filename == original_filename,
                DataCleaningSession.user_id == user_id
            ).first()
            
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset '{original_filename}' not found. Please re-upload the file."
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset not found in memory. Please re-upload and clean the file."
                )
        
        inspection_data = data_utils.get_dataset_inspection(df, original_filename)
        
        # Re-run logic to generate report
        df_temp, status, report_details = data_cleaner.automated_clean_and_report_logic(
            df, original_filename, inspection_data
        )
        
        if status != "Success":
            raise HTTPException(
                status_code=500,
                detail=f"Report generation failed: {report_details.get('key_insight', 'Unknown error')}"
            )
        
        if format.lower() == 'json':
            return JSONResponse(content={"report_details": report_details})
        
        if format.lower() == 'pdf':
            file_stream = data_cleaner.generate_pdf_report(report_details, original_filename)
            
            return StreamingResponse(
                file_stream,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={original_filename}_REPORT.pdf"}
            )
        
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'pdf' or 'json'.")
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

@router.get("/download-data/{file_name}")
async def download_dataset(
    file_name: str,
    format: str = Query("csv", description="Format: 'csv' or 'excel'"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Downloads the cleaned dataset."""
    user_id = current_user.id
    
    # Handle the cleaned filename - remove suffix to get original
    original_filename = file_name.replace('_AUTO_CLEANED', '').replace('_MANUAL_CLEANED', '')
    
    try:
        # First try to get the cleaned version
        cleaned_key = f"{user_id}_{file_name}"
        df = data_utils.global_data_store.get(cleaned_key)
        
        # If not found, try the original file
        if df is None:
            original_key = f"{user_id}_{original_filename}"
            df = data_utils.global_data_store.get(original_key)
        
        # If still not found, check database and raise error
        if df is None:
            session = db.query(DataCleaningSession).filter(
                DataCleaningSession.filename == original_filename,
                DataCleaningSession.user_id == user_id
            ).first()
            
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset '{original_filename}' not found. Please re-upload the file."
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset not found in memory. Please re-upload and clean the file."
                )
        
        if format.lower() == 'csv':
            stream = StringIO()
            df.to_csv(stream, index=False)
            response_content = stream.getvalue().encode('utf-8')
            media_type = "text/csv"
            file_ext = "csv"
        elif format.lower() == 'excel':
            stream = BytesIO()
            df.to_excel(stream, index=False, engine='openpyxl')
            response_content = stream.getvalue()
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_ext = "xlsx"
        else:
            raise HTTPException(status_code=400, detail="Unsupported format.")
        
        return StreamingResponse(
            BytesIO(response_content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={original_filename}_CLEANED.{file_ext}"}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.get("/sessions")
async def get_user_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all cleaning sessions for the current user."""
    sessions = db.query(DataCleaningSession).filter(
        DataCleaningSession.user_id == current_user.id
    ).order_by(DataCleaningSession.upload_date.desc()).all()
    
    return {"sessions": sessions}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a cleaning session."""
    session = db.query(DataCleaningSession).filter(
        DataCleaningSession.id == session_id,
        DataCleaningSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Remove from memory store
    user_file_key = f"{current_user.id}_{session.filename}"
    if user_file_key in data_utils.global_data_store:
        del data_utils.global_data_store[user_file_key]
    
    db.delete(session)
    db.commit()
    
    return {"message": "Session deleted successfully"}