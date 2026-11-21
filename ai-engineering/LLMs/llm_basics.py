from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv(override=True)  # le .env gagne sur les variables Windows

k = os.getenv("GROQ_API_KEY")

client = Groq()


MODEL = "openai/gpt-oss-20b"
def ask(question, temperature=1.0):
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "Tu es un assistant concis. Réponds en francais."},
            {"role": "user", "content": question},
        ],
    )
    print(resp.choices[0].message.content)
    u = resp.usage
    print(f"[tokens entrée={u.prompt_tokens} sortie={u.completion_tokens}]")
    return resp

ask(" cest quoi un llm ?")

