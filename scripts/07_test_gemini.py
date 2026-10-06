import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = "gemini-3.6-flash"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model=MODEL,
    contents="In one short sentence, what is the EPF in Malaysia?",
)
print(response.text)