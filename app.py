from flask import Flask, render_template, request, jsonify
from src.bot import UniAssistBot

app = Flask(__name__)

# Existing UniAssist brain
bot = UniAssistBot()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question."
        })

    try:
        result = bot.get_answer(question)

        return jsonify({
            "answer": result.get(
                "answer",
                "Sorry, mujhe iska answer nahi mila."
            )
        })

    except Exception as error:

        return jsonify({
            "answer": f"Error: {error}"
        })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )