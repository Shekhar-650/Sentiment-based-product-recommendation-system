# app.py
from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    user = None
    error = None
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        if not user:
            error = "Please enter a username."
        else:
            try:
                recs = model.recommend_for_user(user)
                if not recs:
                    error = "No recommendations found for this user. Try a different username."
                else:
                    recommendations = recs
            except Exception as e:
                error = f"Error while computing recommendations: {e}"
    return render_template("index.html", recommendations=recommendations, user=user, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
