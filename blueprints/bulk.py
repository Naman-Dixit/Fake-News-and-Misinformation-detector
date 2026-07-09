"""
blueprints/bulk.py
Mode 2 — Bulk Detection (CSV Upload) REST endpoints.
"""

import csv
import io
import os
import uuid

from flask import Blueprint, request, jsonify, current_app, send_from_directory

from modules.dataset_runner import run_bulk_dataset
from config import allowed_file

bulk_bp = Blueprint("bulk", __name__, url_prefix="/api/bulk")

OUTPUT_CSV_FIELDS = [
    "article_id", "title", "topic", "source", "credibility_score",
    "ensemble_vote", "claim_count", "model_disagreement", "ground_truth", "match",
]


def _resolve_api_credentials(form):
    api_mode = (form.get("api_mode") or "offline").strip().lower()
    api_key = (form.get("api_key") or "").strip()

    if api_mode == "nvidia" and not api_key:
        api_key = current_app.config.get("NVIDIA_API_KEY", "")
    if api_mode == "claude" and not api_key:
        api_key = current_app.config.get("ANTHROPIC_API_KEY", "")

    if api_mode not in ("offline", "nvidia", "claude"):
        api_mode = "offline"

    return api_mode, api_key


@bulk_bp.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Upload a CSV under field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .csv files are supported."}), 400

    api_mode, api_key = _resolve_api_credentials(request.form)

    if api_mode in ("nvidia", "claude") and not api_key:
        return jsonify({"error": f"API mode '{api_mode}' selected but no API key was provided."}), 400

    try:
        raw = file.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(raw))

        required_columns = {"article_id", "title", "text", "source", "topic"}
        if not reader.fieldnames or not required_columns.issubset(set(reader.fieldnames)):
            return jsonify({
                "error": "CSV must contain columns: article_id, title, text, source, topic, ground_truth (optional)."
            }), 400

        rows = list(reader)
        if not rows:
            return jsonify({"error": "CSV file contains no data rows."}), 400

        results, metrics = run_bulk_dataset(rows, api_mode=api_mode, api_key=api_key)
    except UnicodeDecodeError:
        return jsonify({"error": "Could not decode CSV file. Please save it as UTF-8."}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Bulk pipeline execution failed: {exc}"}), 500

    # Write output CSV to disk so it can be downloaded
    output_id = f"results_{uuid.uuid4().hex[:10]}.csv"
    output_path = os.path.join(current_app.config["UPLOAD_FOLDER"], output_id)
    os.makedirs(current_app.config["UPLOAD_FOLDER"], exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in OUTPUT_CSV_FIELDS})

    # Strip the heavy nested pipeline detail before sending back to the client table
    table_results = []
    for r in results:
        row = {k: r.get(k) for k in OUTPUT_CSV_FIELDS}
        row["model_disagreement_value"] = r.get("model_disagreement_value")
        table_results.append(row)

    return jsonify({
        "results": table_results,
        "metrics": metrics,
        "download_id": output_id,
        "row_count": len(results),
        "api_mode": api_mode,
    }), 200


@bulk_bp.route("/download/<path:filename>", methods=["GET"])
def download(filename):
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    safe_name = os.path.basename(filename)
    if not os.path.exists(os.path.join(upload_folder, safe_name)):
        return jsonify({"error": "File not found."}), 404
    return send_from_directory(upload_folder, safe_name, as_attachment=True, download_name="veritas_bulk_results.csv")
