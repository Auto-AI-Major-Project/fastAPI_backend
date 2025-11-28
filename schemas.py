from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

# User schemas
class UserSignup(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    email: Optional[str] = None

# Password change schema
class PasswordChange(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=6)

# User update schema
class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    username: Optional[str] = None
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if v and not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v
    

class UserProfile(BaseModel):
    username: str
    email: str
    profession: str | None = None
    company: str | None = None
    profileImage: str | None = None  # matches frontend naming

    class Config:
        from_attributes = True


class DataCleaningSessionBase(BaseModel):
    filename: str
    original_shape: Optional[str] = None
    cleaned_shape: Optional[str] = None
    cleaning_method: Optional[str] = None
    status: str = "uploaded"

class DataCleaningSessionCreate(DataCleaningSessionBase):
    user_id: int

class DataCleaningSessionResponse(DataCleaningSessionBase):
    id: int
    user_id: int
    upload_date: datetime
    cleaned_date: Optional[datetime] = None
    operations_applied: Optional[str] = None
    
    class Config:
        from_attributes = True

class DataInspectionResponse(BaseModel):
    filename: str
    shape: List[int]
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    preview_first_5_rows: str
    summary_statistics_markdown: str
    message: str

class CleaningOperationRequest(BaseModel):
    operations: List[str]

class CleaningResponse(BaseModel):
    status: str
    message: str
    original_shape: List[int]
    final_shape: List[int]
    cleaned_filename: str
    operations_summary: Optional[Dict[str, str]] = None
    cleaning_summary: Optional[List[str]] = None
    key_insights: Optional[str] = None
    next_steps: Dict[str, str]