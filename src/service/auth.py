import jwt
import os

from dotenv import load_dotenv

load_dotenv()

def sign_jwt(payload: dict) -> str:
    """Sign a JWT with the given payload."""
    return jwt.encode(payload, os.getenv("JWT_SECRET"), algorithm="HS256")