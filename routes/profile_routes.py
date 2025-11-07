from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from database import get_db, User
from auth import get_current_user
from schemas import UserProfile

router = APIRouter(prefix="/api/auth", tags=["Profile"])

@router.get("/profile", response_model=UserProfile)
def get_profile(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "username": user.username,
        "email": user.email,
        "profession": user.profession,
        "company": user.company,
        "profileImage": user.profile_image
    }


@router.put("/profile", response_model=UserProfile)
def update_profile(updated_data: UserProfile, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.username = updated_data.username
    user.email = updated_data.email
    user.profession = updated_data.profession
    user.company = updated_data.company
    user.profile_image = updated_data.profileImage
    user.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(user)
    return {
        "username": user.username,
        "email": user.email,
        "profession": user.profession,
        "company": user.company,
        "profileImage": user.profile_image
    }
