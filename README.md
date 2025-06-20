# Backend for AI Toolbox Hub

This folder contains the FastAPI backend for the AI Toolbox Hub project. Use this folder as the root of your backend deployment repository (e.g., for Render).

## Structure
- `app2.py` — Main FastAPI app
- `tts_cli.py` — CLI utility (if needed)
- `requirements.txt` — Python dependencies
- `speechToText/` — (Optional) Additional scripts or modules

## Deployment
- Set the root directory to `backend` if deploying from a monorepo.
- Use the following start command:
  ```
  uvicorn app2:app --host 0.0.0.0 --port 10000
  ```
- Ensure `requirements.txt` is in this folder. 