import os
import requests

from rag import retrieve_context

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def call_llm(prompt: str) -> str:
    if not API_KEY:
        return "ΛΕΙΠΕΙ OPENROUTER_API_KEY στο .env"
    if not MODEL:
        return "ΛΕΙΠΕΙ OPENROUTER_MODEL στο .env"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful planning agent."},
            {"role": "user", "content": prompt},
        ],
    }

    r = requests.post(url, headers=headers, json=data, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def estimate_study_time(days: int, topics: int) -> str:
    hours_per_topic = 2
    total_hours = topics * hours_per_topic
    per_day = round(total_hours / days, 1)

    return (
        f"Εκτιμώμενος συνολικός χρόνος: {total_hours} ώρες.\n"
        f"Περίπου {per_day} ώρες μελέτης ανά ημέρα."
    )


def run_workflow(user_input: str) -> str:
    # RAG
    context = retrieve_context(user_input, k=4)

    planner_prompt = f"""
Είσαι ο Planner Agent.
Στόχος: να φτιάξεις ΜΙΚΡΟ, ρεαλιστικό πλάνο 4 ημερών.
Χρησιμοποίησε ΜΟΝΟ πληροφορίες που υπάρχουν στο CONTEXT. Αν κάτι δεν υπάρχει, γράψε "γενικό".

Αίτημα: {user_input}

CONTEXT:
{context if context else "Δεν υπάρχει σχετικό context."}

ΕΠΙΣΤΡΟΦΗ: ΜΟΝΟ JSON, χωρίς έξτρα κείμενο, με ακριβώς αυτά τα πεδία:
goal: string (1 πρόταση)
days: array με 4 αντικείμενα, κάθε ένα με:
  day: 1-4
  tasks: array 3-5 σύντομα bullets
questions: array (0-3 σύντομες ερωτήσεις)

ΔΙΑΘΕΣΙΜΟ TOOL:
estimate_study_time(days: int, topics: int)

Αν το αίτημα αφορά πλάνο ημερών/μελέτης και χρειάζεται εκτίμηση χρόνου,
γράψε μέσα στο JSON ένα πεδίο:
use_tool: "estimate_study_time"
και επιπλέον:
days_count: 4
topics_count: 4
"""
    plan = call_llm(planner_prompt)

    # ---- Function Calling (simple) ----
    time_estimation = ""
    if "estimate_study_time" in plan:
        time_estimation = estimate_study_time(days=4, topics=4)

    critic_prompt = f"""
Είσαι ο Critic Agent.
Κοίτα το PLAN και έλεγξε:
1) Αν πατάει στο CONTEXT
2) Αν είναι ρεαλιστικό για 4 μέρες
3) Αν έχει άσχετα/φανταστικά (π.χ. Docker, Gym, KPI) -> ΠΕΤΑ ΤΑ

CONTEXT:
{context if context else "Δεν υπάρχει context."}

PLAN (JSON):
{plan}

ΕΠΙΣΤΡΟΦΗ σε 3 bullets:
- Τι να κοπεί
- Τι να προστεθεί
- Τι να διορθωθεί
"""
    critique = call_llm(critic_prompt)

    executor_prompt = f"""
Είσαι ο Executor Agent.
Δώσε τελικό πλάνο 4 ημερών στα Ελληνικά, ΣΥΝΤΟΜΟ, πρακτικό.
Μη γράψεις πίνακες, μη γράψεις θεωρία, μη γράψεις άσχετα.

CONTEXT:
{context if context else "Δεν υπάρχει context."}

PLAN:
{plan}

CRITIQUE:
{critique}

TOOL OUTPUT (αν υπάρχει):
{time_estimation if time_estimation else "Δεν απαιτήθηκε εκτίμηση χρόνου."}

Μορφή:
Στόχος: ...
Ημέρα 1: 3-5 bullets
Ημέρα 2: 3-5 bullets
Ημέρα 3: 3-5 bullets
Ημέρα 4: 3-5 bullets
Ερωτήσεις (αν υπάρχουν): 0-3 bullets
Εκτίμηση Χρόνου: (βάλε εδώ το tool output αν υπάρχει)
"""
    final_answer = call_llm(executor_prompt)
    return final_answer
