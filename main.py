from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from silero_vad import load_silero_vad, read_audio, save_audio, get_speech_timestamps, collect_chunks
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from piper.voice import PiperVoice
from langdetect import detect
import torch
import wave
import os
import time
import shutil
from tempfile import NamedTemporaryFile
from openai import OpenAI
import json
app = FastAPI()

# Load configurations from environment variables
sampling_rate = int(os.getenv("SAMPLING_RATE", 16000))
OPENAI_API_URL = os.getenv("OPENAI_API_URL","http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","sk-my-secret-key")
DEVICE = os.getenv("DEVICE","cpu")

# Setup OpenAI client
client = OpenAI(
    base_url=OPENAI_API_URL,
    api_key=OPENAI_API_KEY,
)

# Load Silero VAD model
vad_model = load_silero_vad()

# Load Whisper ASR model
def load_asr_pipe():
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=DEVICE,
    )
    return asr_pipe

# Load TTS models
def load_tts_models():
    eng_model = "./tts_models/en_US-lessac-medium.onnx"
    nl_model = "./tts_models/nl_BE-nathalie-medium.onnx"
    
    if DEVICE == 'cuda':
        nl_voice = PiperVoice.load(nl_model,use_cuda=True)
        eng_voice = PiperVoice.load(eng_model,use_cuda=True)
    else:
        nl_voice = PiperVoice.load(nl_model,use_cuda=False)
        eng_voice = PiperVoice.load(eng_model,use_cuda=False)    
    
    voice_models = {
        'en': eng_voice,
        'nl': nl_voice
    }
    return voice_models


asr_pipe = load_asr_pipe()
voice_models = load_tts_models()

def detect_voice(audio_data: bytes):
    # Write audio bytes to temp file
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file.flush()
        wav = read_audio(temp_audio_file.name, sampling_rate=sampling_rate)
    
    # Perform VAD
    speech_timestamps = get_speech_timestamps(wav, vad_model, return_seconds=False)
    if not speech_timestamps:
        return None
    speech_audio = collect_chunks(speech_timestamps, wav)
    
    # Save speech-only audio to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".wav") as output_audio_file:
        save_audio(output_audio_file.name, speech_audio, sampling_rate=sampling_rate)
        return output_audio_file.name


@app.post("/models")
async def list_models():
    models = [i.dict() for i in client.models.list()]
    return JSONResponse(content={"models": models})
    
@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...), generation_args: str = Form(...) ,model: str = None, ):
    # Read uploaded audio file
    audio_data = await file.read()
    
    # Run VAD to get speech-only audio
    speech_audio_path = detect_voice(audio_data)

    # generation args = 
    generation_args = json.loads(generation_args)
    if speech_audio_path:
        # Transcribe speech-only audio
        transcribed = asr_pipe(speech_audio_path)
        transcription_text = transcribed['text'].strip()

        # Generate response with LLM
        messages = [
            {"role": "user", "content": transcription_text}
        ]

        generated_text = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature= generation_args['temperature'],
            top_p= generation_args['top_p'],
            max_completion_tokens = generation_args['max_completion_tokens']).choices[0].message.content

        # Detect language for TTS
        language = detect(generated_text)
        if language not in voice_models:
            return {"error": "Detected language not supported for TTS"}

        # Generate audio response with TTS
        output_file_path = f"output_{language}.wav"
        with wave.open(output_file_path, "wb") as wav_file:
            voice_models[language].synthesize(generated_text, wav_file)
        
        # Move file to static directory to serve
        static_output_path = f"ui/{output_file_path}"
        shutil.move(output_file_path, static_output_path)
        # Return the generated audio file
        return JSONResponse(content={"user":transcription_text,"assistant": generated_text, "audio_url": f"/{output_file_path}"})
        #return FileResponse(output_file_path, media_type="audio/wav", filename=f"response_{language}.wav")
    else:
        error_msg = "I'm sorry, I did not hear your instruction. Please speak more clearly."        
        # Detect language for TTS
        language = 'en'
        # Generate audio response with TTS
        output_file_path = f"output_{language}.wav"
        with wave.open(output_file_path, "wb") as wav_file:
            voice_models[language].synthesize(error_msg, wav_file)

        static_output_path = f"ui/{output_file_path}"
        shutil.move(output_file_path, static_output_path)
        return JSONResponse(content={"user":'',"assistant": error_msg, "audio_url": f"/{output_file_path}"})


app.mount("/digiderius", app)
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
