"""

    Αgentic workflow με LLMs.

Το σύστημα υλοποιεί δύο βασικές “διαδρομές” (routing):
1) Q/A με RAG (όταν ο χρήστης κάνει ερώτηση γνώσης πάνω σε TXT/PDF)
2) Planning workflow (Planner → Critic → Executor) όταν ο χρήστης ζητά πλάνο/οργάνωση

Επίσης:
- Υποστηρίζεται session memory (τελευταία inputs + τελευταίο plan)
- Υποστηρίζεται ένα function calling demo (estimate_study_time)
"""

import os
import requests

from rag import retrieve_context


def call_llm(prompt: str) -> str:
    """
    Κλήση στο LLM μέσω OpenRouter (chat completions)
    Διαβάζει API key και model από environment variables (.env)
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    if not api_key:
        return "ΛΕΙΠΕΙ OPENROUTER_API_KEY στο .env"
    if not model:
        return "ΛΕΙΠΕΙ OPENROUTER_MODEL στο .env"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],

    }

    r = requests.post(url, headers=headers, json=data, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def estimate_study_time(days: int, topics: int) -> str:
    """
    !Demo function calling

    Δίνει μια εκτίμηση χρόνου μελέτης, ώστε να δείξουμε
    πώς ένα LLM μπορεί να “ζητήσει” χρήση εργαλείου και το αποτέλεσμα
    να ενσωματωθεί στην τελική απάντηση
    """
    hours_per_topic = 2
    total_hours = topics * hours_per_topic
    per_day = round(total_hours / days, 1)

    return (
        f"Εκτιμώμενος συνολικός χρόνος: {total_hours} ώρες.\n"
        f"Περίπου {per_day} ώρες μελέτης ανά ημέρα."
    )


def run_workflow(user_input: str, history: list[str], last_plan: str):
    """
    - Παίρνουμε RAG context από data (TXT/PDF)
    - Αν το αίτημα μοιάζει με ερώτηση γνώσης → απαντάμε άμεσα (RAG Q/A)
    - Αλλιώς → Planner → Critic → Executor για παραγωγή πλάνου

    Επιστρέφουμε:
    - final_answer (string)
    - updated_last_plan (plan ή last_plan, ανάλογα με το route)
    """

    # -------------------------
    # Session memory (state)
    # -------------------------
    recent_history = history[-3:] if history else []
    memory_block = f"""
SESSION MEMORY (τελευταία inputs χρήστη):
{recent_history}

ΤΕΛΕΥΤΑΙΟ PLAN (αν υπάρχει):
{last_plan if last_plan else "Κανένα"}
"""

    # -------------------------
    # RAG: ανάκτηση σχετικού context
    # -------------------------
    context = retrieve_context(user_input, k=4)

    # -------------------------
    # Routing: Q/A vs Planning
    # -------------------------
    # Heuristic: αν το input αρχίζει/περιέχει “ερωτηματικές” λέξεις,
    # το αντιμετωπίζουμε ως ερώτηση γνώσης και όχι ως αίτημα για πλάνο.
    qa_keywords = [
        "τι", "ποια", "ποιες", "πώς", "γιατί", "εξήγησε",
        "σύμφωνα", "αναφέρονται", "απειλές", "ασφάλεια", "iot", "pdf"
    ]

    is_question = any(word in user_input.lower() for word in qa_keywords)
    is_planning_request = any(word in user_input.lower() for word in ["πλάνο", "προγραμμα", "οργάνωσε", "πρόγραμμα", "ημέρ", "βδομάδ"])

    # Αν φαίνεται ξεκάθαρα ερώτηση και όχι “θέλω πλάνο” πάμε σε direct Q/A
    if is_question and not is_planning_request:
        qa_prompt = f"""
{memory_block}

Απάντησε στην ερώτηση χρησιμοποιώντας ΜΟΝΟ το CONTEXT.
- Μην φτιάξεις πλάνο.
- Μην γράψεις ημέρες.
- Αν το CONTEXT δεν έχει την απάντηση, πες: "Δεν βρέθηκε στο αρχείο".

ΕΡΩΤΗΣΗ:
{user_input}

CONTEXT:
{context if context else "Δεν υπάρχει context."}
"""
        answer = call_llm(qa_prompt)
        return answer, last_plan

    # -------------------------
    # Planner Agent
    # -------------------------
    # Δεν “κλειδώνουμε” σταθερό χρονικό ορίζοντα.
    # Αν ο χρήστης ζητήσει X ημέρες/εβδομάδες, ο Planner προσαρμόζεται.
    planner_prompt = f"""
{memory_block}

Είσαι ο Planner Agent.
Στόχος: να φτιάξεις ΜΙΚΡΟ, ρεαλιστικό πλάνο σύμφωνα με το αίτημα του χρήστη.
Αν ο χρήστης ζητά ημέρες/εβδομάδες, χώρισε το πλάνο σε αντίστοιχες ενότητες.
Χρησιμοποίησε ΜΟΝΟ πληροφορίες από το CONTEXT. Αν κάτι δεν υπάρχει, γράψε "γενικό".

Αίτημα:
{user_input}

CONTEXT:
{context if context else "Δεν υπάρχει σχετικό context."}

ΕΠΙΣΤΡΟΦΗ: ΜΟΝΟ JSON με πεδία:
goal: string
plan: array of strings (3-8 bullets)
questions: array (0-3 σύντομες)

ΔΙΑΘΕΣΙΜΟ TOOL:
estimate_study_time(days: int, topics: int)

Αν το αίτημα είναι “πλάνο μελέτης/ημέρες” και χρειάζεται χρόνος,
γράψε στο JSON:
use_tool: estimate_study_time
days: <αριθμός>
topics: <αριθμός>
"""
    plan = call_llm(planner_prompt)

    # -------------------------
    # Function calling (demo)
    # -------------------------
    # Κρατάμε το demo απλό:
    # - Αν ο Planner γράψει "estimate_study_time"τότε τρέχουμε το tool
    time_estimation = ""
    if "estimate_study_time" in plan:

        # Στο πλαίσιο της εργασία  το function calling υλοποιείται σε απλοποιημένη μορφή
        # Το LLM επιστρέφει ελεύθερο κείμενο και όχι αυστηρά δομημένο JSON
        # επομένως χρησιμοποιούνται σταθερές παράμετροι (days, topics)
        # ώστε να παρουσιαστεί η έννοια της κλήσης εργαλείου χωρίς επιπλέον πολυπλοκότητα parsing.
        time_estimation = estimate_study_time(days=4, topics=4)

    # -------------------------
    # Critic Agent
    # -------------------------
    critic_prompt = f"""
{memory_block}

Είσαι ο Critic Agent.
Έλεγξε το πλάνο:
- Είναι ρεαλιστικό;
- Πατάει πάνω στο CONTEXT;
- Έχει άσχετα/φανταστικά που πρέπει να κοπούν;

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

    # -------------------------
    # Executor Agent
    # -------------------------
    executor_prompt = f"""
{memory_block}

Είσαι ο Executor Agent.
Δώσε τελικό αποτέλεσμα σύντομο και πρακτικό, σύμφωνα με το αίτημα του χρήστη.
Αν ο χρήστης ζήτησε ημέρες/εβδομάδες, χώρισε το output αντίστοιχα.
Αλλιώς, δώσε bullet plan.

CONTEXT:
{context if context else "Δεν υπάρχει context."}

PLAN:
{plan}

CRITIQUE:
{critique}

TOOL OUTPUT:
{time_estimation if time_estimation else "Δεν απαιτήθηκε εκτίμηση χρόνου."}

Μορφή:
Στόχος: ...
Πλάνο:
- ...
- ...
(αν ζητήθηκαν ημέρες: Ημέρα 1/2/... ή Εβδομάδα 1/2/...)
Εκτίμηση Χρόνου: ...
"""
    final_answer = call_llm(executor_prompt)

    # Επιστρέφουμε final + plan για να ενημερωθεί το last_plan (session state)
    return final_answer, plan
