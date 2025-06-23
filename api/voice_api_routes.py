"""Voice Command API Extensions for ZBAR System.

Endpoints are mounted under the ``/voice`` prefix.  The primary route for
processing a spoken or typed command is ``POST /voice/command``.
"""

from typing import Optional, Dict, Any, List

import speech_recognition as sr
from enhanced_menu_system import EnhancedMenuSystem
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/voice", tags=["voice"])


class VoiceCommand(BaseModel):
    text: str
    source: Optional[str] = "text"  # text, audio_file, microphone


class VoiceResponse(BaseModel):
    status: str
    parsed_tag: Dict[str, Any]
    action: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[Dict]] = None
    message: Optional[str] = None


# Initialize enhanced menu system
menu_system = EnhancedMenuSystem(config={})


@router.post("/command", response_model=VoiceResponse)
async def process_voice_command(command: VoiceCommand):
    """Process voice command and execute appropriate action"""
    try:
        result = menu_system.process_voice_command(command.text)

        return VoiceResponse(
            status=result["status"],
            parsed_tag=result["tag"],
            action=result.get("action"),
            result=result.get("result"),
            suggestions=result.get("suggestions"),
            message=result.get("message")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe")
async def transcribe_audio(audio_file: bytes):
    """Transcribe audio file to text"""
    try:
        recognizer = sr.Recognizer()

        # Save audio temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file)
            tmp_path = tmp.name

        # Transcribe
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Process the transcribed text
        result = menu_system.process_voice_command(text)

        return {
            "transcribed_text": text,
            "processing_result": result
        }

    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Could not understand audio")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples")
async def get_voice_examples():
    """Get voice command examples"""
    return menu_system.get_voice_menu()


@router.get("/history")
async def get_voice_history(limit: int = 10):
    """Get recent voice command history"""
    history = menu_system.voice_history[-limit:]
    return {
        "count": len(history),
        "commands": [tag.__dict__ for tag in history]
    }
