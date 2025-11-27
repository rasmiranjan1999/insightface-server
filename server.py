# server.py
import os
import json
import uuid
import base64
import traceback
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from insightface.app import FaceAnalysis

# -----------------------------
# Config
# -----------------------------
EMB_DIR = "embeddings"
IMG_DIR = "images"
HISTORY_FILE = "history.json"
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Face engine config
MODEL_NAME = "antelope"   # insightface backbone; antelope / buffalo / etc. (antelope is light & accurate)
DET_SIZE = (640, 640)
# Cosine similarity threshold: higher -> stricter. Tweak 0.35..0.45 for best results.
COSINE_THRESHOLD = 0.40

# -----------------------------
# Start FaceAnalysis
# -----------------------------
print("Loading InsightFace models (this may take a moment)...")
fa = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=0, det_size=DET_SIZE)
print("InsightFace ready.")

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__, static_folder="html")
CORS(app)

# -----------------------------
# Helpers
# -----------------------------
def save_history(entry):
    try:
        data = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception:
        data = []
    data.insert(0, entry)
    data = data[:500]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def decode_base64_to_bgr(b64str):
    try:
        if "," in b64str:
            b64str = b64str.split(",", 1)[1]
        b = base64.b64decode(b64str)
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)
        return img
    except Exception:
        traceback.print_exc()
        return None

def get_primary_face_embedding(bgr_image):
    """
    Returns (embedding (np.array 512), bbox dict {x,y,w,h}) or (None, None)
    """
    if bgr_image is None:
        return None, None
    # convert to RGB for insightface
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = fa.get(rgb)
    if not faces:
        return None, None
    # choose the face with largest bbox area
    faces_sorted = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    f = faces_sorted[0]
    emb = f.embedding  # numpy array (512)
    # bbox: [x1, y1, x2, y2]
    x1, y1, x2, y2 = [int(v) for v in f.bbox]
    w = x2 - x1
    h = y2 - y1
    bbox = {"x": int(x1), "y": int(y1), "w": int(w), "h": int(h)}
    return emb, bbox

def cosine_similarity(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# -----------------------------
# Endpoints
# -----------------------------
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json(force=True)
        images = data.get("images") or data.get("image")
        if not images:
            return jsonify({"status":"error","msg":"NO_IMAGE"}), 400
        if not isinstance(images, list):
            images = [images]

        registered_any = False
        for b64 in images:
            bgr = decode_base64_to_bgr(b64)
            if bgr is None:
                continue
            emb, bbox = get_primary_face_embedding(bgr)
            if emb is None:
                continue

            # check duplicates
            duplicate = False
            for fn in os.listdir(EMB_DIR):
                if not fn.endswith(".npy"):
                    continue
                ex_emb = np.load(os.path.join(EMB_DIR, fn))
                sim = cosine_similarity(emb, ex_emb)
                if sim >= COSINE_THRESHOLD:
                    duplicate = True
                    break
            if duplicate:
                continue

            uid = str(uuid.uuid4())[:8]
            img_name = f"{uid}.jpg"
            emb_name = f"{uid}.npy"
            cv2.imwrite(os.path.join(IMG_DIR, img_name), bgr)
            np.save(os.path.join(EMB_DIR, emb_name), emb)
            registered_any = True

        if registered_any:
            return jsonify({"status":"success","msg":"FACE_REGISTERED"})
        else:
            return jsonify({"status":"error","msg":"ALREADY_REGISTERED_OR_NO_FACE"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","msg":"REGISTER_FAILED","error":str(e)}), 500

@app.route("/compare", methods=["POST"])
def compare():
    try:
        data = request.get_json(force=True)
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"status":"error","msg":"NO_IMAGE"}), 400

        bgr = decode_base64_to_bgr(img_b64)
        if bgr is None:
            return jsonify({"status":"error","msg":"INVALID_IMAGE"}), 400

        query_emb, query_bbox = get_primary_face_embedding(bgr)
        if query_emb is None:
            return jsonify({"status":"error","msg":"NO_FACE_DETECTED"}), 400

        best_sim = -1.0
        best_file = None
        best_emb = None
        # compare
        for fn in os.listdir(EMB_DIR):
            if not fn.endswith(".npy"):
                continue
            emb = np.load(os.path.join(EMB_DIR, fn))
            sim = cosine_similarity(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_file = fn
                best_emb = emb

        matched_image = None
        matched_bbox = None
        result_msg = "FACE_NOT_MATCHED"
        if best_sim >= COSINE_THRESHOLD and best_file is not None:
            matched_image = best_file.replace(".npy", ".jpg")
            matched_path = os.path.join(IMG_DIR, matched_image)
            if os.path.exists(matched_path):
                matched_bgr = cv2.imread(matched_path)
                _, matched_bbox = get_primary_face_embedding(matched_bgr)
            result_msg = "FACE_MATCHED"

        # history
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "msg": result_msg,
            "similarity": best_sim if best_file else None,
            "matched_image": matched_image,
            "query_bbox": query_bbox,
            "matched_bbox": matched_bbox
        }
        save_history(entry)

        return jsonify({
            "status":"success",
            "msg": result_msg,
            "image": matched_image,
            "similarity": best_sim if best_file else None,
            "query_bbox": query_bbox,
            "matched_bbox": matched_bbox
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","msg":"COMPARE_FAILED","error":str(e)}), 500

@app.route("/debug_compare", methods=["POST"])
def debug_compare():
    """
    Accepts JSON: {"img1": base64, "img2": base64}
    Returns cosine similarity (debug)
    """
    try:
        data = request.get_json(force=True)
        img1 = decode_base64_to_bgr(data.get("img1"))
        img2 = decode_base64_to_bgr(data.get("img2"))
        e1, _ = get_primary_face_embedding(img1)
        e2, _ = get_primary_face_embedding(img2)
        if e1 is None or e2 is None:
            return jsonify({"error":"face_not_found"}), 400
        sim = cosine_similarity(e1, e2)
        return jsonify({"similarity": sim})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}), 500

@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())

@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return jsonify([])
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# Serve UI & images
@app.route("/")
def home():
    return send_from_directory("html", "server.html")

@app.route("/<path:filename>")
def static_files(filename):
    # allow /images/<file>
    if filename.startswith("images/") or filename.startswith("embeddings/"):
        return send_from_directory(".", filename)
    return send_from_directory("html", filename)

# -----------------------------
# Run server for Render deployment 
# -----------------------------
if __name__ == "__main__":
    import os
    PORT = int(os.environ.get("PORT", 5000))  # use Render's assigned port or default to 5000
    print(f"Starting Flask server on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)

# if __name__ == "__main__":
#     print("Starting Flask server on http://0.0.0.0:5000")
#     app.run(host="0.0.0.0", port=5000, debug=False)
