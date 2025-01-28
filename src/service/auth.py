import jwt
import os

from dotenv import load_dotenv

load_dotenv()

def sign_jwt(payload: dict) -> str:
    """Sign a JWT with the given payload."""
    secret = os.getenv("JWT_SIGNING_SECRET")
    if not secret:
        raise ValueError("JWT_SIGNING_SECRET not set in environment variables. Please set it.")
    return jwt.encode(payload, secret, algorithm="HS256")