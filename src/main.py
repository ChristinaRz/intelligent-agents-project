"""

Entry point (CLI) του συστήματος ευφυών πρακτόρων.

Η main υλοποιεί τα εξής:
- Φορτώνει το εικονικό περιβάλλον ( env)
- Διαχειρίζεται την αλληλεπίδραση με τον χρήστη μέσω γραμμής εντολών
- Διατηρεί απλό session state (history + last_plan)
- Καλεί το agentic workflow και εμφανίζει το αποτέλεσμα

"""

from dotenv import load_dotenv
from agents import run_workflow


def main():
    """
    Κύρια συνάρτηση εκτέλεσης του CLI.

    - διαβάζει input από τον χρήστη
    - ενημερώνει το session state
    - καλεί το run_workflow()
    - εμφανίζει την απάντηση
    """
    # Φόρτωση μεταβλητών περιβάλλοντος (API keys, model κ.λπ.)
    load_dotenv()

    print("LLM Generic Planner. Γράψε 'exit' για έξοδο.")

    # Session history: κρατάμε τα προηγούμενα inputs του χρήστη
    history = []

    #αποθηκεύει το τελευταίο παραγόμενο πλάνο
    # και επιτρέπει στον χρήστη να το τροποποιήσει σε επόμενα prompts
    last_plan = ""

    while True:
        # Ανάγνωση input χρήστη
        user_input = input("> ").strip()

        # Τερματισμός προγράμματος
        if user_input.lower() == "exit":
            break

        # Αγνόηση κενών inputs
        if not user_input:
            continue

        # Ενημέρωση session history
        history.append(user_input)

        # Εκτέλεση agentic workflow:
        # - επιστρέφει τελική απάντηση
        # - ενημερώνει το last_plan αν παρήχθη νέο πλάνο
        result, last_plan = run_workflow(
            user_input=user_input,
            history=history,
            last_plan=last_plan
        )

        # Εμφάνιση αποτελέσματος στον χρήστη
        print("\nΑΠΟΤΕΛΕΣΜΑ:")
        print(result)
        print("-" * 30)


# Standard Python entry point
if __name__ == "__main__":
    main()
