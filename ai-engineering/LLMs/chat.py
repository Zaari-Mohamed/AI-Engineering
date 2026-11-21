from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv(override=True)

client = Groq()
MODEL = "openai/gpt-oss-20b"
PRIX_ENTREE, PRIX_SORTIE = 0.075, 0.30  # $ par million de tokens : à vérifier sur groq.com/pricing !

history = [{"role": "system", "content": "Tu es un assistant concis. Réponds en français."}]
total_in, total_out = 0, 0

print("Chat (tape 'exit' pour quitter)")

while True:
    user = input("\nToi > ")
    if user.strip().lower() in {"exit", "quit"}:
        break

    history.append({"role": "user", "content": user})
    print("IA > ", end="", flush=True)

    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=history,
        stream=True,
    )

    reply = ""
    usage = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
            reply += text
        # Groq envoie les stats d'usage dans le dernier chunk
        if getattr(chunk, "x_groq", None) and chunk.x_groq.usage:
            usage = chunk.x_groq.usage

    history.append({"role": "assistant", "content": reply})

    if usage:
        total_in += usage.prompt_tokens
        total_out += usage.completion_tokens

cout = total_in / 1e6 * PRIX_ENTREE + total_out / 1e6 * PRIX_SORTIE
print(f"\nTokens : {total_in} entrée / {total_out} sortie -> ~{cout:.4f} $")