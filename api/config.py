import os

MONGODB_URL = os.getenv("MONGODB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with open('../label_prompt.txt', 'r') as file:
    LABEL_READER_PROMPT = file.read()
