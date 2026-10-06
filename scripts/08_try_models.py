import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for model_name in CANDIDATES:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with the single word: ready",
        )
        print("WORKS ", model_name, "->", response.text.strip())
    except Exception as error:
        print("FAILED", model_name, "->", str(error)[:100])
    time.sleep(2)