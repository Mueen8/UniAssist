"""
train.py - Trains the UniAssist intent classifier.

Pipeline:
  Dataset -> load -> split train/test -> preprocess -> TF-IDF
          -> train classifier -> evaluate -> retrain on all data -> save

Run with:  python train.py
"""

from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.config import MODEL_FILE, RANDOM_STATE, TEST_SIZE
from src.data_loader import load_intents
from src.preprocessing import clean_text


def build_vectorizer():
    # TF-IDF over single words AND pairs of words (e.g. "tuition fee")
    return TfidfVectorizer(ngram_range=(1, 2))


def build_classifier():
    # Logistic Regression: simple, fast, and gives probabilities
    # (we need probabilities for the "confidence" check).
    return LogisticRegression(C=10, max_iter=1000)


def main():
    # ---------- STEP 1: load data ----------
    print("STEP 1: Loading dataset...")
    texts, labels, _ = load_intents()
    print(f"  Total example questions: {len(texts)}")
    print(f"  Number of intents      : {len(set(labels))}")
    for intent, count in sorted(Counter(labels).items()):
        print(f"    {intent:25s} {count} questions")

    # ---------- STEP 2: split into training and test data ----------
    # The model learns from the training part only.
    # The test part is "new" questions used to check honestly how good it is.
    print("\nSTEP 2: Splitting into training and test data...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,  # keep a fair share of every intent in both parts
    )
    print(f"  Training questions: {len(train_texts)}")
    print(f"  Test questions    : {len(test_texts)}")

    # ---------- STEP 3: preprocess ----------
    print("\nSTEP 3: Cleaning the text...")
    train_clean = [clean_text(t) for t in train_texts]
    test_clean = [clean_text(t) for t in test_texts]
    print(f"  Example: '{train_texts[0]}'  ->  '{train_clean[0]}'")

    # ---------- STEP 4: text -> numbers (TF-IDF) ----------
    print("\nSTEP 4: Converting text to numbers with TF-IDF...")
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_clean)  # learn vocabulary + weights
    X_test = vectorizer.transform(test_clean)        # only apply (never learn!)
    print(f"  Vocabulary size (number of features): {X_train.shape[1]}")

    # ---------- STEP 5: train the classifier ----------
    print("\nSTEP 5: Training the classifier...")
    classifier = build_classifier()
    classifier.fit(X_train, train_labels)
    print("  Training finished.")

    # ---------- STEP 6: evaluate on the unseen test questions ----------
    print("\nSTEP 6: Evaluating on test questions...")
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"  Test accuracy: {accuracy * 100:.1f}%\n")
    print(classification_report(test_labels, predictions, zero_division=0))

    wrong = [
        (q, true, pred)
        for q, true, pred in zip(test_texts, test_labels, predictions)
        if true != pred
    ]
    if wrong:
        print("  Questions the model got wrong:")
        for question, true, pred in wrong:
            print(f"    '{question}'  (correct: {true}, predicted: {pred})")
    else:
        print("  The model got every test question right.")

    # ---------- STEP 7: retrain on ALL data and save ----------
    # The test above told us the approach works. For the final chatbot we
    # use every question we have, so it learns as much as possible.
    print("\nSTEP 7: Retraining on ALL data and saving the model...")
    final_clean = [clean_text(t) for t in texts]
    final_vectorizer = build_vectorizer()
    X_all = final_vectorizer.fit_transform(final_clean)
    final_classifier = build_classifier()
    final_classifier.fit(X_all, labels)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": final_vectorizer, "classifier": final_classifier},
        MODEL_FILE,
    )
    print(f"  Model saved to: {MODEL_FILE}")
    print("\nDone! Now run:  python chatbot.py")


if __name__ == "__main__":
    main()
