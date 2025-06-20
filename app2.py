"""FastAPI server wrapper for the Tiny CLI Text‑to‑Speech tool
----------------------------------------------------------------
• GET  /       – Web interface for easy testing
• POST /tts    – Convert text → speech (WAV), requires X-API-Key header.
• GET  /key    – Show the auto‑generated API key *once* (optional, can be removed).
• GET  /health – Simple health‑check endpoint.

By default, an API key is generated on first run and stored in the file
`.api_key`.  You can also set the env‑var `API_KEY` beforehand to use a fixed
key instead.

Run locally:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import uuid
from tempfile import NamedTemporaryFile
from pathlib import Path
import tempfile
import shutil
import gc
from faster_whisper import WhisperModel
import subprocess

import pyttsx3
from fastapi import FastAPI, Header, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# ─── CONFIG ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _init_api_key() -> str:
    """Return API key, loading from env or persistent file, or creating new."""
    # 1️⃣ Environment variable wins
    env_key = os.getenv("API_KEY")
    if env_key:
        return env_key

    # 2️⃣ Try persistent file (keeps key stable across restarts)
    path = ".api_key"
    if os.path.exists(path):
        with open(path, "r", encoding="utf‑8") as f:
            return f.read().strip()

    # 3️⃣ Generate new and persist
    new_key = uuid.uuid4().hex
    with open(path, "w", encoding="utf‑8") as f:
        f.write(new_key)
    print(f"[INFO] Generated new API key → {new_key} (saved to {path})")
    return new_key

API_KEY = _init_api_key()

# ---------------------------------------------------------------------------
# ─── APP -------------------------------------------------------------------
# ---------------------------------------------------------------------------

app = FastAPI(title="TTS‑Engine", version="1.0.0", swagger_ui_parameters={"defaultModelsExpandDepth": -1})

# Optional: enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# ─── UTILITIES -------------------------------------------------------------
# ---------------------------------------------------------------------------

def _verify_key(header_key: str | None) -> None:
    """Raise 401 if key absent or invalid."""
    if header_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


def _synthesize(text: str, rate: int = 100) -> bytes:
    """Render WAV bytes using pyttsx3 and return them."""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)

    with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        engine.save_to_file(text, tmp.name)
        engine.runAndWait()
        tmp.seek(0)
        return tmp.read()

# ---------------------------------------------------------------------------
# ─── WEB INTERFACE ---------------------------------------------------------
# ---------------------------------------------------------------------------

HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text-to-Speech API</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            resize: vertical;
            font-size: 16px;
            font-family: inherit;
            transition: border-color 0.3s ease;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s ease;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            background: #cccccc;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            font-weight: 500;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .info {
            background-color: #cce7ff;
            color: #004085;
            border: 1px solid #99d3ff;
        }
        .api-info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 4px solid #667eea;
        }
        .api-info h3 {
            margin-top: 0;
            color: #667eea;
        }
        audio {
            width: 100%;
            margin-top: 15px;
        }
        .download-btn {
            background: #28a745;
            margin-top: 10px;
        }
        .endpoint-list {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .endpoint-list li {
            margin-bottom: 8px;
        }
        .endpoint-list a {
            color: #667eea;
            text-decoration: none;
        }
        .endpoint-list a:hover {
            text-decoration: underline;
        }
        pre {
            background: #f4f4f4;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Text-to-Speech API</h1>
        
        <div class="api-info">
            <h3>🚀 Service Information</h3>
            <p><strong>Status:</strong> <span style="color: #28a745;">●</span> Online and Ready</p>
            <p><strong>API Key:</strong> <code>""" + API_KEY + """</code></p>
            <p><strong>Base URL:</strong> <code>""" + "http://31.220.75.241:9001" + """</code></p>
        </div>

        <form id="ttsForm">
            <div class="form-group">
                <label for="textInput">✍️ Enter text to convert to speech:</label>
                <textarea id="textInput" rows="4" placeholder="Type or paste your text here..." required></textarea>
            </div>
            
            <div class="form-group">
                <label for="rateInput">🎚️ Speech Rate (60-200 WPM):</label>
                <input type="range" id="rateInput" min="60" max="200" value="100" style="width: 100%;">
                <span id="rateValue">100 WPM</span>
            </div>
            
            <button type="submit" id="convertBtn">🔊 Convert to Speech</button>
        </form>

        <div id="status"></div>
        <div id="audioResult"></div>

        <div class="endpoint-list">
            <h3>📡 Available Endpoints</h3>
            <ul>
                <li><strong>Text-to-Speech:</strong> POST /tts</li>
                <li><strong>Health Check:</strong> <a href="/health" target="_blank">GET /health</a></li>
                <li><strong>API Key:</strong> <a href="/key" target="_blank">GET /key</a></li>
                <li><strong>API Docs:</strong> <a href="/docs" target="_blank">GET /docs</a></li>
            </ul>
        </div>

        <div style="margin-top: 25px;">
            <h3>💻 cURL Example</h3>
            <pre>curl -X POST "http://31.220.75.241:9001/tts" \\
  -H "X-API-Key: """ + API_KEY + """" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello world!", "rate": 100}' \\
  --output speech.wav</pre>
        </div>
    </div>

    <script>
        const API_KEY = '""" + API_KEY + """';
        const rateInput = document.getElementById('rateInput');
        const rateValue = document.getElementById('rateValue');
        
        rateInput.addEventListener('input', function() {
            rateValue.textContent = this.value + ' WPM';
        });

        document.getElementById('ttsForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const textInput = document.getElementById('textInput');
            const convertBtn = document.getElementById('convertBtn');
            const status = document.getElementById('status');
            const audioResult = document.getElementById('audioResult');
            const rate = parseInt(rateInput.value);
            
            if (!textInput.value.trim()) {
                showStatus('Please enter some text to convert.', 'error');
                return;
            }
            
            convertBtn.disabled = true;
            convertBtn.textContent = '🔄 Converting...';
            showStatus('Converting text to speech...', 'info');
            
            try {
                const response = await fetch('/tts', {
                    method: 'POST',
                    headers: {
                        'X-API-Key': API_KEY,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: textInput.value.trim(),
                        rate: rate
                    })
                });
                
                if (response.ok) {
                    const audioBlob = await response.blob();
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    audioResult.innerHTML = `
                        <h4>✅ Audio Generated Successfully!</h4>
                        <audio controls autoplay>
                            <source src="${audioUrl}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                        <br>
                        <a href="${audioUrl}" download="speech.wav">
                            <button type="button" class="download-btn">💾 Download Audio</button>
                        </a>
                    `;
                    
                    showStatus('Text converted to speech successfully!', 'success');
                } else {
                    const errorText = await response.text();
                    showStatus(`Error: ${response.status} - ${errorText}`, 'error');
                }
            } catch (error) {
                showStatus(`Network error: ${error.message}`, 'error');
            } finally {
                convertBtn.disabled = false;
                convertBtn.textContent = '🔊 Convert to Speech';
            }
        });
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        // Test API health on page load
        window.addEventListener('load', async function() {
            try {
                const response = await fetch('/health');
                if (response.ok) {
                    showStatus('✅ API is online and ready to use!', 'success');
                } else {
                    showStatus('⚠️ API is not responding properly.', 'error');
                }
            } catch (error) {
                showStatus('❌ Cannot connect to API. Service may be down.', 'error');
            }
        });
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# ─── ENDPOINTS -------------------------------------------------------------
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Web"])
async def root():
    """Web interface for the Text-to-Speech API."""
    return HTML_INTERFACE

@app.post("/tts", responses={200: {"content": {"audio/wav": {}}}}, tags=["TTS"])
async def tts(
    text: str,
    rate: int = 100,
    x_api_key: str | None = Header(default=None),
):
    """Convert **text** to speech (WAV).

    **Headers**:
    • `X-API-Key`: your secret key (string)

    **Query params**:
    • `text` (str): text to convert (required)
    • `rate` (int, optional): 60–200 words/min (default 100)
    """
    _verify_key(x_api_key)

    if not (60 <= rate <= 200):
        raise HTTPException(status_code=400, detail="Rate must be 60–200 WPM")

    audio_bytes = _synthesize(text, rate)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")


@app.get("/key", response_class=JSONResponse, tags=["Admin"])
async def show_key():
    """Return the current API key *once* as a convenience.
    ⚠️ Consider disabling this in production!"""
    return {"api_key": API_KEY}


@app.get("/health", tags=["Health"])
async def health():
    """Health‑check endpoint."""
    return {"status": "ok"}

@app.post("/transcribe", tags=["STT"])
async def transcribe(
    file: UploadFile,
    x_api_key: str | None = Header(default=None),
    device: str = "cpu"
):
    """Transcribe an audio or video file to text using Whisper."""
    _verify_key(x_api_key)
    # Save uploaded file to a temp location
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        input_path = tmpdir / file.filename
        with open(input_path, "wb") as f:
            f.write(await file.read())
        # Extract audio to wav (mono, 16kHz)
        def extract_audio(video: Path, wav: Path, duration: int | None = None):
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                   "-ac", "1", "-ar", "16000", "-vn"]
            if duration:
                cmd += ["-t", str(duration)]
            cmd.append(str(wav))
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        full_wav = tmpdir / "audio_full.wav"
        sample_wav = tmpdir / "audio_30s.wav"
        extract_audio(input_path, full_wav)
        extract_audio(input_path, sample_wav, 30)
        # Detect language
        tiny = WhisperModel("tiny", device=device, compute_type="int8")
        segments, info = tiny.transcribe(str(sample_wav), language=None, beam_size=1)
        lang = info.language or "en"
        tiny = None
        gc.collect()
        # Remove tiny model cache
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        for p in cache.glob("**/openai--whisper-tiny*"):
            shutil.rmtree(p, ignore_errors=True)
        # Transcribe full audio
        model_id = "small.en" if lang == "en" else "small"
        small = WhisperModel(model_id, device=device, compute_type="int8")
        segments, _ = small.transcribe(str(full_wav), language=lang, beam_size=5, vad_filter=True)
        transcript = " ".join(s.text.strip() for s in segments)
    return {"transcript": transcript, "language": lang}

# ---------------------------------------------------------------------------
# ─── Uvicorn Entry‑Point ---------------------------------------------------
# ---------------------------------------------------------------------------

# Allows:  `python app.py`  for quick local dev (use gunicorn/uvicorn in prod)
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)), reload=True)
