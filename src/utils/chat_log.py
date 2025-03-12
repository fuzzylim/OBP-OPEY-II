import os
from supabase import create_client, Client
from datetime import datetime

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def log_chat_message(message: str):
    """Log a chat message to Supabase."""
    # Ensure no PII is logged
    sanitized_message = sanitize_message(message)
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": sanitized_message
    }
    supabase.table("chat_logs").insert(data).execute()


def sanitize_message(message: str) -> str:
    """Sanitize the message to remove PII."""
    # Implement PII removal logic here
    return message
