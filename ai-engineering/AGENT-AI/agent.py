# agent.py : agent minimal avec 3 outils (Groq)
import ast, json, operator
from datetime import datetime
from rag import retrieve, ingest, llm, MODEL
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)                       # nombres négatifs
    if isinstance(node, ast.BinOp):
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("expression non autorisée")

def calculator(expression: str) -> str:
    return str(_eval(ast.parse(expression, mode="eval").body))

def search_docs(query: str) -> str:
    docs, metas = retrieve(query, k=3)
    return "\n".join(f"[{m['source']}] {d}" for d, m in zip(docs, metas))

def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

TOOL_FUNCS = {"calculator": calculator, "search_docs": search_docs,
              "current_time": current_time}

def tool(name, desc, props, required):
    # format Groq / OpenAI : "function" + "parameters"
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object",
                                        "properties": props,
                                        "required": required}}}

TOOLS = [
    tool("calculator", "Évalue une expression arithmétique, ex: '4*25'.",
         {"expression": {"type": "string"}}, ["expression"]),
    tool("search_docs", "Cherche dans les documents RH (congés, "
         "télétravail, notes de frais).", {"query": {"type": "string"}}, ["query"]),
    tool("current_time", "Donne la date et l'heure actuelles.", {}, []),
]
SYSTEM = (
    "Tu es un assistant RH. Règles :\n"
    "1. Les messages contiennent souvent des fautes de frappe : comprends l'intention "
    "(ex. « cnge » = congé) et formule des recherches propres, en français correct.\n"
    "2. Les règles de l'entreprise (nombre de jours de congés par an, report, "
    "télétravail, plafonds de remboursement, délais) se trouvent dans les documents. "
    "Ne demande JAMAIS à l'utilisateur une information qui peut être dans les "
    "documents : cherche-la d'abord avec search_docs.\n"
    "3. Pour tout calcul, utilise l'outil calculator.\n"
    "4. N'invente jamais de chiffres ni d'exemples. Si l'information n'est pas dans "
    "les documents, dis-le clairement.\n"
    "5. Ne demande à l'utilisateur que ce qui est personnel et ne peut pas être dans "
    "les documents (ex. combien de jours il a déjà pris).\n"
    "\n"
    "Exemple : « j'ai pris 8 jours de congé, combien il me reste ? »\n"
    "→ search_docs(\"nombre de jours de congés par an\") → 25 jours\n"
    "→ calculator(\"25-8\") → 17\n"
    "→ « Il vous reste 17 jours (25 par an, moins 8 pris). »"
)
def run_agent(goal, max_steps=6):
    messages = [{"role": "user", "content": goal}, {"role": "system", "content": SYSTEM}]
    for step in range(max_steps):
        resp = llm.chat.completions.create(
            model=MODEL, max_tokens=1000, temperature=0,
            tools=TOOLS, tool_choice="auto", messages=messages)
        msg = resp.choices[0].message

        if not msg.tool_calls:                          # réponse finale
            return msg.content or ""

        # on remet le message de l'assistant (avec ses appels d'outils) dans l'historique
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}") or {}
            except json.JSONDecodeError:
                args = {}
            print(f"  [étape {step + 1}] outil {tc.function.name}({args})")
            try:
                out = TOOL_FUNCS[tc.function.name](**args)
            except Exception as e:                      # erreur renvoyée au LLM
                out = f"Erreur : {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": str(out)})
    return "Arrêt : nombre maximal d'étapes atteint."

if __name__ == "__main__":
    ingest()                      # indexe les documents avant de démarrer
    while True:
        q = input("\nObjectif > ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        print(run_agent(q))