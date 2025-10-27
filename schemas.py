from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime

# User Schemas
class SignUpRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    confirm_password: str = Field(..., alias="confirmPassword")
    profile_image: Optional[str] = Field(None, alias="profileImage")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class UpdateProfileRequest(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    profession: Optional[str] = None
    company: Optional[str] = None
    profile_image: Optional[str] = Field(None, alias="profileImage")

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    profession: Optional[str] = None
    company: Optional[str] = None
    profile_image: Optional[str] = Field(None, alias="profileImage")
    
    class Config:
        from_attributes = True
        populate_by_name = True

class AuthResponse(BaseModel):
    message: str
    token: str
    user: Optional[UserResponse] = None

class MessageResponse(BaseModel):
    message: str