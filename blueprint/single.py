"""
blueprints/single.py
Mode 1 — Single Article Detection REST endpoint.
"""

from flask import Blueprint, request, jsonify, current_app

from modules.claim_extractor import extract_claims
from modules.corroboration import verify_claims
from modules.semantic_analysis import analyze_semantics
from modules.ensemble_engine import run_ensemble

single_bp = Blueprint("single", __name__, url_prefix="/api/single")


def _resolve_api_credentials(payload):
    api_mode = (payload.get("api_mode") or "offline").strip().lower()
    api_key = (payload.get("api_key") or "").strip()

    if api_mode == "nvidia" and not api_key:
        api_key = current_app.config.get("NVIDIA_API_KEY", "")
    if api_mode == "claude" and not api_key:
        api_key = current_app.config.get("ANTHROPIC_API_KEY", "")

    if api_mode not in ("offline", "nvidia", "claude"):
        api_mode = "offline"

    return api_mode, api_key


@single_bp.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}

    title = (payload.get("title") or "").strip()
    text = (payload.get("text") or "").strip()
    source = (payload.get("source") or "").strip()
    topic = (payload.get("topic") or "").strip()

    if not title or not text or not source:
        return jsonify({
            "error": "Missing required fields. 'title', 'text', and 'source' are required."
        }), 400

    api_mode, api_key = _resolve_api_credentials(payload)

    if api_mode in ("nvidia", "claude") and not api_key:
        return jsonify({
            "error": f"API mode '{api_mode}' selected but no API key was provided."
        }), 400

    try:
        claims = extract_claims(title, text, source, topic, api_mode=api_mode, api_key=api_key)
        verifications = verify_claims(claims, source, topic, api_mode=api_mode, api_key=api_key)
        semantics = analyze_semantics(title, text, api_mode=api_mode, api_key=api_key)
        result = run_ensemble(claims, verifications, semantics, source)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Pipeline execution failed: {exc}"}), 500

    result["article"] = {
        "title": title,
        "source": source,
        "topic": topic or "General",
    }
    result["api_mode"] = api_mode

    return jsonify(result), 200
