# Truck Detection Backend

Python backend service that receives camera uploads over FTP, runs YOLO-based truck detection/classification, uploads media to S3, and stores detection metadata in Supabase.

## Features

- Runs an FTP server for inbound camera files.
- Detects and tracks trucks in uploaded videos.
- Classifies truck bin state (`empty` / `full`) using a second YOLO model.
- Infers movement direction (`incoming` / `outgoing`) from tracking history.
- Uploads videos and detection images to S3.
- Writes video and truck-detection records to Supabase.

## Tech Stack

- Python
- `pyftpdlib` (FTP server)
- `ultralytics` + OpenCV (YOLO inference/video processing)
- `boto3` (S3 upload)
- `supabase` (database writes)

## Project Structure

- `FTP Server.py` - main application entrypoint.
- `requirements.txt` - Python dependencies.
- `.env` - runtime configuration (not safe for source control).
- `first_step_model.pt` - YOLO model for detection/tracking (expected in repo root).
- `second_step_model.pt` - YOLO model for classification (expected in repo root).

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Access to a Supabase project with required tables:
  - `video`
  - `camera`
  - `truck`
  - `truck_detections`
- S3 bucket with upload permissions.
- YOLO model files:
  - `first_step_model.pt`
  - `second_step_model.pt`
- OS-level permission for FTP port `21` (requires root/admin on Linux).

## Setup

1. Clone and enter the project directory.
2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add model weights to the repository root:
   - `first_step_model.pt`
   - `second_step_model.pt`

## Environment Variables

Create/update `.env`:

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# AWS / S3
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
REGION_NAME=us-east-1
ENDPOINT_URL=https://s3.us-east-1.amazonaws.com
BUCKET_NAME=your_bucket_name
```

Important:
- Do not commit real credentials.
- Rotate any credentials that were accidentally exposed.

## FTP Configuration

The FTP server user and upload directory are currently hardcoded in `FTP Server.py`:

- Username: `reolink`
- Password: `H@rryP0tter`
- Upload root: `C:\reolink`
- Bind address/port: `0.0.0.0:21`

If running on Linux, update the upload path to a valid Linux path (for example `/var/reolink`) and ensure the process has permission to read/write that directory and bind to port `21`.

## Run

```bash
python "FTP Server.py"
```

Server output should indicate:
- Supabase client initialization (if configured)
- FTP server startup
- File reception and analysis pipeline logs

## Processing Flow

1. Camera uploads a file via FTP.
2. If file is a video (`.mp4`, `.avi`, `.mov`, `.mkv`):
   - Upload original video to S3
   - Insert video row in Supabase (`video` table)
   - Run truck detection/tracking + bin classification
   - Save best-frame image locally
   - Upload image to S3
   - Insert detection row in Supabase (`truck_detections`)
3. Optional local cleanup removes generated image after successful insert.

## Notes

- The detection threshold and class IDs are configured in `FTP Server.py`.
- This app currently runs as a long-lived process and does not expose an HTTP API.
- Ensure table schemas in Supabase match expected fields used by the code.

## Troubleshooting

- **Model not found**: confirm `.pt` files exist in project root.
- **Cannot bind FTP port 21**: run with elevated privileges or use a non-privileged port (e.g. `2121`) and update camera settings.
- **S3 upload failures**: verify AWS credentials, region, bucket name, and IAM permissions.
- **Supabase insert failures**: verify service role key and table/column names.
- **Cannot open video**: validate file upload completed and file path is accessible.
