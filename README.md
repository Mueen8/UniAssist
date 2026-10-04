# UniAssist - University Student Assistant (Version 1)

A chatbot that answers student questions using our own dataset and a trained
machine-learning model (TF-IDF + Logistic Regression). No external AI APIs.

## Run (Windows CMD, inside the UniAssist folder)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python train.py
    python chatbot.py

## Update the university information

1. Open `data/intents.json`
2. Replace the [SAMPLE] responses, add or edit example questions
3. Run `python train.py` again
4. Run `python chatbot.py`

## Files

- data/intents.json      knowledge: questions + answers per intent
- src/config.py          settings and file paths
- src/preprocessing.py   text cleaning
- src/data_loader.py     reads intents.json
- src/bot.py             UniAssistBot class (loads model, answers questions)
- train.py               trains, evaluates and saves the model
- chatbot.py             command-line chat
