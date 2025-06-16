from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

GROQ = os.getenv("GROQ")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")

client = Groq(api_key=GROQ)

def converse_with_llm(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a movie recommendation assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"LLM ERROR: {e}")
        raise