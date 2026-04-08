# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Production-grade FastAPI server for Nutrisync RL Environment.

Exposes environment over HTTP/WebSockets via the OpenEnv standard interface.
Unified with Gradio UI for Hugging Face Spaces.
"""
import gradio as gr
import sys
import os
import logging
import traceback
from fastapi import Request, FastAPI, Response
from fastapi.responses import RedirectResponse, JSONResponse
from openenv.core.env_server.http_server import create_app

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Root of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import Environment
# Note: server/environment.py now correctly inherits from openenv.core.base.Environment
try:
    from server.environment import NutrisyncEnv as NutrisyncEnvironment
    logger.info("Successfully imported NutrisyncEnvironment from server.environment")
except ImportError as e:
    logger.error(f"Import Error (primary): {e}")
    try:
        from environment import NutrisyncEnv as NutrisyncEnvironment
    except ImportError as e2:
        logger.error(f"Import Error (secondary): {e2}")
        from NutriSync.server.environment import NutrisyncEnv as NutrisyncEnvironment

# Import Models
try:
    from models import NutrisyncAction, NutrisyncObservation
except ImportError:
    from NutriSync.models import NutrisyncAction, NutrisyncObservation

# Import Gradio UI
try:
    import app as gradio_app
    ui = gradio_app.ui
except ImportError:
    sys.path.append(PROJECT_ROOT)
    import app as gradio_app
    ui = gradio_app.ui

# Create standard OpenEnv FastAPI application
# Standard routes (/reset, /step, /health, /info) will stay at root
app = create_app(NutrisyncEnvironment, NutrisyncAction, NutrisyncObservation)

# --- Enhanced Diagnostics (Temporary) ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_details = {
        "error": str(exc),
        "type": type(exc).__name__,
        "traceback": traceback.format_exc(),
        "path": request.url.path,
        "method": request.method
    }
    logger.error(f"CRASH in {request.url.path}: {error_details['error']}\n{error_details['traceback']}")
    return JSONResponse(status_code=500, content=error_details)

# --- HTTPS / Proxy Fix for Mixed Content ---

@app.middleware("http")
async def https_proxy_middleware(request: Request, call_next):
    """
    Ensure that we respect the X-Forwarded-Proto header from Hugging Face's proxy.
    This helps Gradio generate the correct interior URLs (https instead of http).
    """
    if request.headers.get("x-forwarded-proto") == "https":
        # We can't easily change request.url.scheme as it's immutable in some versions,
        # but we can set it in the scope if needed.
        request.scope["scheme"] = "https"
    
    response = await call_next(request)
    return response

# Mount Gradio UI
# path="/gradio" matches the HF Blank Page Error iframe target
app = gr.mount_gradio_app(app, ui, path="/gradio")

@app.get("/")
def index():
    """Redirect root to the Gradio UI."""
    return RedirectResponse(url="/gradio")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    # proxy_headers=True is critical for HF Spaces
    uvicorn.run("server.app:app", host=host, port=port, proxy_headers=True, forwarded_allow_ips="*")

if __name__ == '__main__':
    main()
