from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")