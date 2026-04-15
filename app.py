from flask import Flask, render_template, request
from utils import extract_text_from_pdf, calculate_score, suggest_role

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        role = request.form["role"]
        file = request.files["resume"]

        resume_text = extract_text_from_pdf(file)
        score = calculate_score(resume_text, role)
        suggestion = suggest_role(resume_text)

        return render_template(
            "result.html",
            score=score,
            role=role,
            suggestion=suggestion
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
