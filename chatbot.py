"""
chatbot.py - Command-line interface for UniAssist.

Run with:  python chatbot.py
Type 'quit' or 'exit' to stop.
"""

from src.bot import UniAssistBot

# Set to False to hide the "[intent ... confidence ...]" line
SHOW_DEBUG = True


def main():
    bot = UniAssistBot()

    print("=" * 60)
    print("  UniAssist - University Student Assistant")
    print("  Ask me about admissions, fees, exams and more.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUniAssist: Goodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("UniAssist: Goodbye!")
            break

        result = bot.get_answer(question)

        if SHOW_DEBUG:
            print(
                f"  [intent: {result['intent']} | "
                f"confidence: {result['confidence']:.2f}]"
            )
        print(f"UniAssist: {result['answer']}")


if __name__ == "__main__":
    main()
