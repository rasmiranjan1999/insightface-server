# ----------------------------------------------------
# server.py – InsightFace Server with Auto Model Download
# ----------------------------------------------------
import os
import json
import uuid
import base64
import traceback
from datetime import datetime

import numpy as np
import cv2
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from insightface.app import FaceAnalysis

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
EMB_DIR = "embeddings"
IMG_DIR = "images"
MODEL_DIR = "models/antelope"
HISTORY_FILE = "history.json"

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URLS = {
    f"{MODEL_DIR}/1k3d68.onnx":
        "https://github.com/rasmiranjan1999/insightface-server/releases/download/v1-models/1k3d68.onnx",
    f"{MODEL_DIR}/2d106det.onnx":
        "https://github.com/rasmiranjan1999/insightface-server/releases/download/v1-models/2d106det.onnx",
    f"{MODEL_DIR}/genderage.onnx":
        "https://github.com/rasmiranjan1999/insightface-server/releases/download/v1-models/genderage.onnx",
    f"{MODEL_DIR}/glintr100.onnx":
        "https://github.com/rasmiranjan1999/insightface-server/releases/download/v1-models/glintr100.onnx",
    f"{MODEL_DIR}/scrfd_10g_bnkps.onnx":
        "https://github.com/rasmiranjan1999/insightface-server/releases/download/v1-models/scrfd_10g_bnkps.onnx",
}

COSINE_THRESHOLD = 0.40
DET_SIZE = (640, 640)
MODEL_NAME = "antelope"

# ----------------------------------------------------
# MODEL DOWNLOAD
# ----------------------------------------------------
def download_model(url, save_path):
    if os.path.exists(save_path):
        print(f"[OK] Model exists: {save_path}")
        return

    print(f"[DOWNLOAD] {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)

    print(f"[DONE] Saved: {save_path}")

# download all models
for path, url in MODEL_URLS.items():
    download_model(url, path)

# ----------------------------------------------------
# LOAD INSIGHTFACE
# ----------------------------------------------------
print("Loading InsightFace models...")
fa = FaceAnalysis(name=MODEL_NAME, root="models", providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=0, det_size=DET_SIZE)
print("InsightFace ready!")

# ----------------------------------------------------
# FLASK APP
# ----------------------------------------------------
app = Flask(__name__, static_folder="html")
CORS(app)

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def save_history(entry):
    try:
        data = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
    except:
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
    except:
        traceback.print_exc()
        return None


def get_primary_face_embedding(bgr_image):
    if bgr_image is None:
        return None, None

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = fa.get(rgb)

    if not faces:
        return None, None

    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True
    )
    f = faces_sorted[0]

    emb = f.embedding
    x1, y1, x2, y2 = map(int, f.bbox)
    bbox = {"x": x1, "y": y1, "w": x2-x1, "h": y2-y1}
    return emb, bbox


def cosine_similarity(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ----------------------------------------------------
# API: REGISTER
# ----------------------------------------------------
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json(force=True)
        images = data.get("images") or data.get("image")

        if not images:
            return jsonify({"status": "error", "msg": "NO_IMAGE"}), 400
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
                if fn.endswith(".npy"):
                    old_emb = np.load(os.path.join(EMB_DIR, fn))
                    if cosine_similarity(emb, old_emb) >= COSINE_THRESHOLD:
                        duplicate = True
                        break
            if duplicate:
                continue

            uid = str(uuid.uuid4())[:8]
            img_path = f"{uid}.jpg"
            emb_path = f"{uid}.npy"

            cv2.imwrite(os.path.join(IMG_DIR, img_path), bgr)
            np.save(os.path.join(EMB_DIR, emb_path), emb)

            registered_any = True

        if registered_any:
            return jsonify({"status": "success", "msg": "FACE_REGISTERED"})
        return jsonify({"status": "error", "msg": "ALREADY_REGISTERED_OR_NO_FACE"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "msg": "REGISTER_FAILED", "error": str(e)}), 500

# ----------------------------------------------------
# API: COMPARE
# ----------------------------------------------------
@app.route("/compare", methods=["POST"])
def compare():
    try:
        data = request.get_json(force=True)
        img_b64 = data.get("image")

        if not img_b64:
            return jsonify({"status": "error", "msg": "NO_IMAGE"}), 400

        bgr = decode_base64_to_bgr(img_b64)
        if bgr is None:
            return jsonify({"status": "error", "msg": "INVALID_IMAGE"}), 400

        query_emb, query_bbox = get_primary_face_embedding(bgr)
        if query_emb is None:
            return jsonify({"status": "error", "msg": "NO_FACE_DETECTED"}), 400

        best_sim = -1.0
        best_file = None

        for fn in os.listdir(EMB_DIR):
            if fn.endswith(".npy"):
                emb = np.load(os.path.join(EMB_DIR, fn))
                sim = cosine_similarity(query_emb, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_file = fn

        matched_image = None
        matched_bbox = None
        result_msg = "FACE_NOT_MATCHED"

        if best_sim >= COSINE_THRESHOLD and best_file:
            matched_image = best_file.replace(".npy", ".jpg")
            matched_bgr = cv2.imread(os.path.join(IMG_DIR, matched_image))

            if matched_bgr is not None:
                _, matched_bbox = get_primary_face_embedding(matched_bgr)

            result_msg = "FACE_MATCHED"

        # history
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "msg": result_msg,
            "similarity": best_sim,
            "matched_image": matched_image,
            "query_bbox": query_bbox,
            "matched_bbox": matched_bbox
        }
        save_history(entry)

        return jsonify({
            "status": "success",
            "msg": result_msg,
            "image": matched_image,
            "similarity": best_sim,
            "query_bbox": query_bbox,
            "matched_bbox": matched_bbox
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "msg": "COMPARE_FAILED", "error": str(e)}), 500

# ----------------------------------------------------
# API: DEBUG COMPARE
# ----------------------------------------------------
@app.route("/debug_compare", methods=["POST"])
def debug_compare():
    try:
        data = request.get_json(force=True)
        img1 = decode_base64_to_bgr(data.get("img1"))
        img2 = decode_base64_to_bgr(data.get("img2"))

        e1, _ = get_primary_face_embedding(img1)
        e2, _ = get_primary_face_embedding(img2)

        if e1 is None or e2 is None:
            return jsonify({"error": "face_not_found"}), 400

        return jsonify({"similarity": cosine_similarity(e1, e2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------
# API: HISTORY
# ----------------------------------------------------
@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return jsonify([])

# ----------------------------------------------------
# STATIC FILES
# ----------------------------------------------------
@app.route("/")
def home():
    return send_from_directory("html", "server.html")


@app.route("/<path:filename>")
def static_files(filename):
    if filename.startswith("images/") or filename.startswith("embeddings/"):
        return send_from_directory(".", filename)
    return send_from_directory("html", filename)

# ----------------------------------------------------
# RUN SERVER
# ----------------------------------------------------
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
# ----------------------------------------------------