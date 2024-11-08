import argparse
import os
import uvicorn

def parse_args():
    parser = argparse.ArgumentParser(description="Start the FastAPI server with custom settings.")
    # Define command-line arguments
    parser.add_argument("--port", type=int, default=8080, help="Port to run the FastAPI server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to run the FastAPI server on")
    parser.add_argument("--OPENAI_API_URL", type=str, default="http://127.0.0.1:8000/v1", help="The URL where a OpenAI compatilble model server is hosted.")
    parser.add_argument("--OPENAI_API_KEY", type=str, default="sk-my-secret-key", help="The key to use for connecting to the OpenAI server")
    parser.add_argument("--sampling_rate", type=int, default=16000, help="Sampling rate for audio processing")
    parser.add_argument("--DEVICE", type=str, default='cuda', help="The device to run models on ('cpu' or 'cuda')")
    return parser.parse_args()

def main():
    args = parse_args()
    # Set environment variables for the FastAPI app
    os.environ["SAMPLING_RATE"] = str(args.sampling_rate)
    os.environ["OPENAI_API_URL"] = args.OPENAI_API_URL
    os.environ["OPENAI_API_KEY"] = args.OPENAI_API_KEY
    os.environ["DEVICE"] = args.DEVICE
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()
