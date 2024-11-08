# Digiderius
TTS and STT. Uses an OpenAI compatible server (vLLM, OpenAI, https://github.com/njelicic/openai-server, etc.) to generate the LLM response.  

## Install requirements
```bash
git clone https://github.com/erasmus-university-library/digiderius
cd digiderius
pip3 install -r  requirements.txt
```

## Get STT models:
```
cd digiderius
mkdir tts_models
wget -i ../downloads.txt
```

## Start server
```bash
python3 cli.py \
--host=127.0.0.1 \
--port=8080 \
--OPENAI_API_URL="http://127.0.0.1:8000/v1" \ 
--OPENAI_API_KEY="sk-my-secret-key"
```

Next, navigate to http://127.0.0.1:8080/ and start talking. 
