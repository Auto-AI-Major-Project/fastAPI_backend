from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from database import get_db, User
from auth import (
    get_password_hash, authenticate_user, create_access_token,
    get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from schemas import UserSignup, UserLogin, UserResponse, Token, PasswordChange, UserUpdate

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/signup")
async def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail={"message": "Email already registered"})
    
    existing_username = db.query(User).filter(User.username == user_data.username).first()
    if existing_username:
        raise HTTPException(status_code=400, detail={"message": "Username already taken"})
    
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name or user_data.username,
        is_active=True,
        is_verified=False
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(
        data={"sub": new_user.email, "user_id": new_user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {
        "token": access_token,
        "user": {"id": new_user.id, "email": new_user.email, "username": new_user.username, "full_name": new_user.full_name},
        "message": "Account created successfully"
    }


@router.post("/signin")
async def signin(user_credentials: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail={"message": "Incorrect email or password"})
    if not user.is_active:
        raise HTTPException(status_code=403, detail={"message": "Inactive user account"})

    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {
        "token": access_token,
        "user": {"id": user.id, "email": user.email, "username": user.username, "full_name": user.full_name},
        "message": "Signed in successfully"
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    return current_user


@router.put("/update-profile", response_model=UserResponse)
async def update_profile(user_update: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    if user_update.full_name:
        current_user.full_name = user_update.full_name

    if user_update.username:
        existing = db.query(User).filter(User.username == user_update.username, User.id != current_user.id).first()
        if existing:
            raise HTTPException(status_code=400, detail={"message": "Username already taken"})
        current_user.username = user_update.username

    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/change-password")
async def change_password(password_data: PasswordChange, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    from auth import verify_password
    if not verify_password(password_data.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail={"message": "Incorrect old password"})

    current_user.hashed_password = get_password_hash(password_data.new_password)
    current_user.updated_at = datetime.utcnow()
    db.commit()
    return {"message": "Password updated successfully"}


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    return {"message": "Logged out successfully"}
