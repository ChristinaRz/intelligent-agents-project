from dotenv import load_dotenv
load_dotenv()

from agents import run_workflow

print("LLM Generic Planner. Γράψε 'exit' για έξοδο.")

while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break

    result = run_workflow(user_input)
    print("\nΑΠΟΤΕΛΕΣΜΑ:")
    print(result)
    print("-" * 30)
