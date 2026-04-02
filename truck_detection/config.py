import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

FIRST_MODEL_PATH = "first_step_model.pt"
SECOND_MODEL_PATH = "second_step_model.pt"
TRUCK_CLASS_ID = [0, 1, 2, 3, 4]
MIN_CONF = 0.7
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = "truck_detections"

supabase: Client | None
if not SUPABASE_URL or not SUPABASE_KEY:
    print("!!! WARNING: Supabase URL or KEY not found in .env !!!")
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[Supabase] Client initialized")
