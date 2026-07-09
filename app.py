"""
app.py
Fake News & Misinformation Detector — Flask application entry point.

Two modes only:
    1. Single Article Detection
    2. Bulk Detection (CSV Upload)

Run:
    python app.py
"""

import os
from flask import Flask, render_template

from config import Config
from blueprints.single import single_bp
from blueprints.bulk import bulk_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    app.register_blueprint(single_bp)
    app.register_blueprint(bulk_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        return {"status": "ok", "service": "Fake News & Misinformation Detector"}

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
