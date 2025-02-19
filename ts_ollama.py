import os
import psycopg2

from ollama import chat
from ollama import ChatResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

try:
    conn = psycopg2.connect(**db_config)
    print("Connected to the database successfully!")
except Exception as e:
    print("Error connecting to the database:", e)

response: ChatResponse = chat(model='deepseek-r1:1.5b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)