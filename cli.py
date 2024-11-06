import argparse
import os
import uvicorn

def parse_args():
    parser = argparse.ArgumentParser(description="Start the FastAPI server with custom settings.")
    
    # Define command-line arguments
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token for authentication")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to run the FastAPI server on")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name for LLM")
    parser.add_argument("--sampling_rate", type=int, default=16000, help="Sampling rate for audio processing")

    return parser.parse_args()

def main():
    args = parse_args()

    # Set environment variables for the FastAPI app
    os.environ["HF_TOKEN"] = args.hf_token
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["SAMPLING_RATE"] = str(args.sampling_rate)

    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()
