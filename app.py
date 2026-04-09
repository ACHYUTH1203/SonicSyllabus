
import os
import re
import asyncio
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from state import AudioRequest
from dotenv import load_dotenv
from openai import OpenAI
from graph import build_and_run_graph

load_dotenv()
app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

USER_REQUESTS = defaultdict(int)
SUPPORTED_LANGUAGES = ["english", "hindi", "telugu", "tamil", "kannada", "malayalam", "marathi", "bengali", "gujarati", "punjabi", "odia", "assamese", "urdu"]

def slugify(text: str):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

def generate_tts_audio(script_text: str, filepath: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.audio.speech.create(model="tts-1-hd", voice="nova", input=script_text)
    response.stream_to_file(filepath)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/generate")
async def generate_audio(req: AudioRequest):
    if USER_REQUESTS[req.user_id] >= 5:
        raise HTTPException(status_code=429, detail="Daily limit reached")

    target_language = req.language.lower()
    if target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")
        
    topic_slug = slugify(req.topic)
    cached_filename = f"revision_bite_{target_language}_{topic_slug}.mp3"
    cached_filepath = os.path.join("static", cached_filename)
    text_filepath = cached_filepath.replace(".mp3", ".txt")
    
    # Cache Hit Logic (Now returns text + audio)
    if os.path.exists(cached_filepath):
        print(f"\nCACHE HIT: {cached_filename}")
        await asyncio.sleep(2) # Shorter artificial delay for better UX
        USER_REQUESTS[req.user_id] += 1
        
        cached_text = ""
        if os.path.exists(text_filepath):
            with open(text_filepath, "r", encoding="utf-8") as f:
                cached_text = f.read()
                
        return {"status": "success", "audio_url": f"/static/{cached_filename}", "text": cached_text}

    print(f"\nRunning LangGraph Pipeline for: {req.topic}...")
    final_state = build_and_run_graph(req.topic, req.language.title())
    
    if not final_state.get("is_valid"):
        raise HTTPException(status_code=400, detail=final_state.get("rejection_message"))

    final_script = final_state.get("translated_script", final_state.get("english_script", ""))
    
    try:
        # Generate Audio
        generate_tts_audio(final_script, cached_filepath)
        
        # Cache Text
        with open(text_filepath, "w", encoding="utf-8") as f:
            f.write(final_script)
            
        USER_REQUESTS[req.user_id] += 1
        return {"status": "success", "audio_url": f"/static/{cached_filename}", "text": final_script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\nStarting SuperKalam Web Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)