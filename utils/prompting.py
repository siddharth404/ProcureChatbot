from typing import List, Dict, Tuple

SYSTEM_BASE = "You are ProcureCopilot, an assistant for infrastructure tender analysis. Answer with compliance-focused, accurate, and actionable insights."

def build_messages(query: str, retrieved, response_mode: str, web_snippets = None):
    web_snippets = web_snippets or []
    context_blocks = []
    for score, chunk in retrieved:
        context_blocks.append(f"[score={score:.3f}] {chunk}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "No local context."

    web_block = ""
    if web_snippets:
        joined = "\n".join([f"- {w['title']} | {w['link']}\n  {w.get('snippet','')}" for w in web_snippets])
        web_block = f"\n\n[Web Results]\n{joined}"

    if response_mode == "Concise":
        user_directive = "Provide a short, bullet summary with key numbers and requirements. If uncertain, clearly state assumptions."
    else:
        user_directive = "Provide a detailed, well-structured response with headers, bullet points, and explicit citations to [Local #i] or [Web #i] where relevant."

    system = f"""{SYSTEM_BASE}
[Local Context]
{context_text}
{web_block}
Follow domain: Indian EPC tenders (MoRTH/CPWD/Metro etc.). Always be precise (units, rupees, dates).
"""

    user = f"""{user_directive}

Query: {query}"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages
