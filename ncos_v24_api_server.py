#!/usr/bin/env python3
"""
ncOS v24 API Server - FINAL
This is the correct API server that connects to the ncos_v24_final_engine.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

# Lifespan management
engine_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    global engine_instance
    print("🚀 Initializing ncOS v24 Engine for API...")
    try:
        from ncos_v24_final_engine import ncOSFinalEngine
        engine_instance = ncOSFinalEngine()
        print("✅ ncOS v24 Engine loaded successfully within API.")
    except Exception as e:
        print(f"🔥 FATAL ERROR: Could not initialize ncos_v24_final_engine: {e}")
        engine_instance = None

    yield  # The API is now running

    # Code to run on shutdown
    print("🛑 Shutting down API server.")
    if engine_instance:
        engine_instance.checkpoint("System shutdown.")
    print("✅ Session state saved. Goodbye.")

# FastAPI Application
app = FastAPI(
    title="ncOS v24 Trading API",
    description="The API gateway for the persistent ZanFlow trading engine.",
    version="24.0",
    lifespan=lifespan
)

class CommandRequest(BaseModel):
    command: str

# API Endpoints
@app.get("/")
def read_root():
    """Root endpoint to confirm the server is running."""
    return {"status": "ncOS v24 API is online", "version": "24.0"}

@app.post("/command", response_model=Dict[str, Any])
async def process_command_endpoint(request: CommandRequest):
    """A single, unified endpoint to process any command."""
    if not engine_instance:
        raise HTTPException(status_code=503, detail="Engine is not available.")

    try:
        result = engine_instance.process_command(request.command)
        return result
    except Exception as e:
        logging.error(f"Error processing command '{request.command}': {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.post("/scan", response_model=Dict[str, Any])
async def scan_market_endpoint():
    """Endpoint specifically for market scanning."""
    return await process_command_endpoint(CommandRequest(command="scan"))

@app.post("/decide", response_model=Dict[str, Any])
async def decide_trade_endpoint():
    """Endpoint specifically for making a trading decision."""
    return await process_command_endpoint(CommandRequest(command="decide"))

@app.get("/status", response_model=Dict[str, Any])
async def get_status_endpoint():
    """Endpoint to get the current system status."""
    if not engine_instance:
        raise HTTPException(status_code=503, detail="Engine is not available.")
    return engine_instance.get_system_status()
