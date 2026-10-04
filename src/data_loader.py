"""
data_loader.py - Reads data/intents.json.

Returns:
  texts     -> list of example questions        ["How can I apply?", ...]
  labels    -> intent of each question          ["admission_process", ...]
  responses -> dictionary intent -> answer      {"admission_process": "..."}
"""

import json

from src.config import DATA_FILE, FALLBACK_INTENT, MIN_QUESTIONS_PER_INTENT


def load_intents(path=DATA_FILE):
    # Read the JSON file
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = []
    labels = []
    responses = {}

    for item in data["intents"]:
        intent = item["intent"]
        questions = item["questions"]

        # Basic safety checks so mistakes give a clear message
        if len(questions) < MIN_QUESTIONS_PER_INTENT:
            raise ValueError(
                f"Intent '{intent}' has only {len(questions)} questions. "
                f"Please add at least {MIN_QUESTIONS_PER_INTENT}."
            )

        responses[intent] = item["response"]

        for question in questions:
            texts.append(question)
            labels.append(intent)

    if FALLBACK_INTENT not in responses:
        raise ValueError(f"Your data must contain a '{FALLBACK_INTENT}' intent.")

    return texts, labels, responses
