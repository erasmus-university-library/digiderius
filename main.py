from fastapi import FastAPI, UploadFile, File, Request
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

app = FastAPI()

# Load configurations from environment variables
hf_token = os.getenv("HF_TOKEN")
model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
sampling_rate = int(os.getenv("SAMPLING_RATE", 16000))

# Load Silero VAD model
vad_model = load_silero_vad()

# Load Whisper ASR model

def load_asr_pipe():
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device="cpu",
    )
    return asr_pipe

# Load TTS models
def load_tts_models():
    eng_model = "./tts_models/en_US-lessac-medium.onnx"
    eng_voice = PiperVoice.load(eng_model,use_cuda=False)

    nl_model = "./tts_models/nl_BE-nathalie-medium.onnx"
    nl_voice = PiperVoice.load(nl_model,use_cuda=False)

    voice_models = {
        'en': eng_voice,
        'nl': nl_voice
    }
    return voice_models

# Load LLM model pipeline
def load_llm_pipe():
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.bfloat16,
                                             bnb_4bit_use_double_quant=True,
                                             bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config,
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    pipe = pipeline(
        "text-generation", 
        model=model,
        tokenizer=tokenizer
    )
    return pipe

generation_args = { 
    "max_new_tokens": 512, 
    "return_full_text": False, 
    "temperature": 1.0, 
    "do_sample": True, 
    "repetition_penalty": 1.2
} 

llm_pipe = load_llm_pipe()
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

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    # Read uploaded audio file
    audio_data = await file.read()
    
    # Run VAD to get speech-only audio
    speech_audio_path = detect_voice(audio_data)
    if speech_audio_path:
        # Transcribe speech-only audio
        transcribed = asr_pipe(speech_audio_path)
        transcription_text = transcribed['text'].strip()

        # Generate response with LLM
        messages = [
            {"role": "user", "content": transcription_text}
        ]
        
        answer = llm_pipe(messages, **generation_args)[0]
        generated_text = answer['generated_text']
        
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
