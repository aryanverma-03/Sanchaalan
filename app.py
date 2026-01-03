import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import mvp  

UPLOAD_FOLDER = "uploads"
SNAPSHOT_FOLDER = "snapshots_out"
ALLOWED_EXTENSIONS = {"pdf", "docx", "eml"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SNAPSHOT_FOLDER"] = SNAPSHOT_FOLDER

# Enable CORS for all routes so React (localhost:3000) can call Flask (localhost:5000)
CORS(app)

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No filename"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) #type: ignore
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Run your Phase-1 pipeline
        outputs = mvp.process_file(file_path)  # should return [(role, path)]
        results = [{"role": r, "filename": os.path.basename(p)} for r, p in outputs]

        return jsonify({"snapshots": results})

    return jsonify({"error": "Unsupported file type"}), 400


@app.route("/snapshots/<path:filename>")
def get_snapshot(filename):
    return send_from_directory(app.config["SNAPSHOT_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
