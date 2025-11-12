from typing import Any, Dict
from guardrails import Guard
from guardrails.hub import ProfanityFree
#from guardrails.hub import ToxicLanguage
import asyncio

# Initialize the Guard with validators directly
guard = Guard().use_many(
    ProfanityFree(on_fail="exception"),
    #ToxicLanguage(threshold=0.7, validation_method="sentence", on_fail="exception")
    #PII
    #Hallucination
    #HateSpeech
)

async def evaluate_input(user_input: str):
    try:
        # Run the guard on the user input
        result =  guard.validate(user_input, return_validated_output=True)
        return {
            "status": "safe",
            "validated_output": result.validated_output,
            "validation_passed": result.validation_passed
        }
    except Exception as e:
        return {
            "status": "unsafe",
            "error": str(e),
            "validation_passed": False
        }

if __name__ == "__main__":
    user_input = input("Enter user input to validate: ")
    result = asyncio.run(evaluate_input(user_input))

    # Print the evaluation result
    print("\nEvaluation Result:")
    for key, value in result.items():
        print(f"{key}: {value}")