"""
config.py - Central settings for UniAssist.
Change values here instead of editing the other files.
"""

from pathlib import Path

# Project root folder (the folder that contains train.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Where our knowledge (questions + answers) lives
DATA_FILE = BASE_DIR / "data" / "intents.json"

# Where the trained model will be saved / loaded from
MODEL_FILE = BASE_DIR / "models" / "uniassist_model.joblib"

# Fraction of the data kept aside to TEST the model (0.2 = 20%)
TEST_SIZE = 0.2

# Fixed number so the train/test split is the same every run
RANDOM_STATE = 42

# If the model is less confident than this (0.0 to 1.0),
# we do not trust it and give the fallback answer instead.
CONFIDENCE_THRESHOLD = 0.25

# Name of the intent used for unknown questions
FALLBACK_INTENT = "fallback"

# Every intent needs at least this many example questions
MIN_QUESTIONS_PER_INTENT = 5
