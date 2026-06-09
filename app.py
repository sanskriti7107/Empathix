from flask import Flask, render_template, request, jsonify
import os, requests, time, logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# HF config (optional)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL = "unitary/unbiased-toxic-roberta"

def hf_infer_toxicity(text):
    """Call HF model once (used for slower/full analysis)."""
    if not HF_API_TOKEN:
        return None
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text, "options": {"wait_for_model": False}}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=6)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        app.logger.debug("HF call failed: %s", e)
    return None

def heuristic_score(text):
    """Fast heuristic scoring for real-time meter (cheap & local)."""
    txt = (text or "").lower()
    if not txt.strip():
        return 0.0
    # weight length, exclamation, bad words
    bad_words = ["stupid","idiot","ugly","hate","dumb","kill","shut up","trash","bitch","slut"]
    score = 0.0
    # presence of bad words
    for w in bad_words:
        if w in txt:
            score += 0.4
    # caps and exclamations amplify
    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    score += min(0.25, caps_ratio * 2.0)
    score += min(0.2, text.count("!") * 0.05)
    # negative sentiment heuristics (simple)
    negative_triggers = ["idiot","hate","stfu","shut up","die"]
    for t in negative_triggers:
        if t in txt:
            score += 0.2
    # normalize to 0..1
    return min(1.0, round(score, 3))

def analyze_text(text):
    """Return label and score. Use HF when available, else heuristic."""
    # Try HF first but fallback fast if HF not available or slow
    hf = hf_infer_toxicity(text)
    if hf and isinstance(hf, list):
        # parse model output (model-specific)
        toxic_score = 0.0
        try:
            for item in hf[0]:
                lbl = item.get("label","").lower()
                s = float(item.get("score",0))
                if any(k in lbl for k in ["toxic","insult","threat","abuse","hate","obscene"]):
                    toxic_score += s
            toxic_score = min(1.0, toxic_score)
            label = "TOXIC" if toxic_score >= 0.25 else "SAFE"
            return {"label": label, "score": round(toxic_score,3), "method":"hf"}
        except Exception as e:
            app.logger.debug("HF parse error: %s", e)
    # fast fallback
    sc = heuristic_score(text)
    label = "TOXIC" if sc >= 0.25 else "SAFE"
    return {"label": label, "score": sc, "method":"heuristic"}

def suggest_rewrite(text):
    # simple suggestion demo — extend later with GPT rewrite
    txt = (text or "").lower()
    if any(w in txt for w in ["stupid","idiot","dumb","hate","ugly"]):
        return "Try: 'I don't agree, but I respect your perspective.'"
    if "!" in text and sum(1 for c in text if c == "!") > 1:
        return "Try calming language: 'I hear you — can you explain more?'"
    return ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/score", methods=["POST"])
def api_score():
    data = request.get_json() or {}
    text = data.get("text","")
    if not text.strip():
        return jsonify({"label":"SAFE","score":0.0})
    res = analyze_text(text)
    return jsonify({"label": res["label"], "score": res["score"], "method": res.get("method","heuristic")})

@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json() or {}
    text = data.get("text","").strip()
    if not text:
        return jsonify({"ok": False, "message":"Empty comment."})
    res = analyze_text(text)
    suggestion = suggest_rewrite(text)
    if res["label"] == "TOXIC":
        logging.info("Blocked: score=%.3f text=%s", res["score"], text)
        return jsonify({
            "ok": False,
            "message": "❌ Toxic comment blocked by Empathix",
            "suggestion": suggestion,
            "score": res["score"]
        })
    return jsonify({"ok": True, "message":"✅ Comment allowed", "score": res["score"]})

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, port=port)