from flask import Flask, request, jsonify
from flask_cors import CORS
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from steps import step1_matrix, step2_gaussian, step3_subspaces
from steps import step4_independence, step5_gramschmidt, step6_projection
from steps import step7_leastsquares, step8_eigenvalues, step9_svd

app = Flask(__name__)
CORS(app)


@app.route("/api/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image uploaded"}), 400

    file_bytes = file.read()

    s1 = step1_matrix.run(file_bytes)
    s2 = step2_gaussian.run(s1["matrix"])
    s3 = step3_subspaces.run(s2["blur_matrix"])
    s4 = step4_independence.run(s2["blur_matrix"])
    s5 = step5_gramschmidt.run(s2["blur_matrix"])
    s6 = step6_projection.run(s2["blur_matrix"], s2["blurred_matrix"])
    s7 = step7_leastsquares.run(s2["blur_matrix"], s2["blurred_matrix"], s1["matrix"])
    s8 = step8_eigenvalues.run(s2["blur_matrix"])
    s9 = step9_svd.run(s2["blur_matrix"], s2["blurred_matrix"], s1["matrix"])

    return jsonify({
        "step1": s1, "step2": s2, "step3": s3,
        "step4": s4, "step5": s5, "step6": s6,
        "step7": s7, "step8": s8, "step9": s9
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
