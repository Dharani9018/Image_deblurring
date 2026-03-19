# Image Deblurring
---

## What this project does
Takes a clean image, applies a known blur using matrix multiplication,
then recovers it using a 9-step linear algebra pipeline.
The entire pipeline is visualised on a website — each step shows
the image/matrix, key numbers, and a plain-English explanation.

---

## Tech stack
- Backend  : Python, Flask, NumPy, SciPy, scikit-image, Matplotlib
- Frontend : HTML, CSS, Vanilla JavaScript
- Comms    : REST API (Flask serves JSON to the frontend)

---


### steps 1–3 + server
- app.py         : Flask server, one POST route /api/process
- utils.py       : shared helpers (image to base64, save plot, etc.)
- step1_matrix   : load image → grayscale → NumPy matrix
- step2_gaussian : build blur kernel matrix, apply blur, compute RREF via sympy, LU decomp
- step3_subspaces: compute rank, nullity, column space basis, null space basis

### steps 4–6
- step4_independence : check linear independence of blur matrix columns, extract basis
- step5_gramschmidt  : run Gram-Schmidt on the basis vectors → orthogonal basis
- step6_projection   : project blurred image vector onto column space of blur matrix

### steps 7–9 
- step7_leastsquares : solve x̂ = (AᵀA)⁻¹Aᵀb using np.linalg.lstsq
- step8_eigenvalues  : compute eigenvalues of AᵀA, plot spectrum
- step9_svd          : SVD of blur matrix, truncated reconstruction at k=5,20,50
### frontend
- index.html         : upload UI + 9 step cards
- style.css          : layout and design
- main.js            : send image to API, receive JSON, render each step card
---

## How each step file is structured
Every step file exports exactly one function.
It takes input (matrix or image), does the math, returns a dict.
app.py imports all 9 and chains them.

Example:
  step9_svd.py  →  def run(blur_matrix, blurred_image):
                       ...
                       return { "k5": img, "k20": img, "k50": img, "singular_values": [...] }

---

## What each step returns (sent to frontend as JSON)

Step 1 : original image (base64), pixel matrix preview (first 4×4), shape
Step 2 : blurred image (base64), RREF matrix preview, rank, nullity
Step 3 : rank, nullity, column space basis vectors, null space basis vectors
Step 4 : independent columns list, basis matrix
Step 5 : orthogonal basis vectors (before vs after Gram-Schmidt)
Step 6 : projected image (base64), projection matrix preview
Step 7 : least squares recovered image (base64), PSNR score
Step 8 : eigenvalue list, spectrum plot (base64)
Step 9 : k=5 image, k=20 image, k=50 image (all base64), PSNR + SSIM per k

---

## What the website shows per step card
- Step number + title  (e.g. "Step 2 — Blur matrix + RREF")
- Pipeline stage badge (e.g. "Matrix simplification")
- Image or matrix visual
- Key metric chips     (rank, nullity, PSNR, etc.)
- Insight bar          (one plain-English sentence explaining what just happened)

---

## How to run locally
  cd backend
  pip install -r requirements.txt
  python app.py               # starts on localhost:5000

  open frontend/index.html    # in any browser
  # OR
  cd frontend
  python -m http.server 3000  # serves on localhost:3000

---

## Resume line (after finishing)
A full-stack image deblurring web app implementing a 9-step
linear algebra pipeline (RREF, Gram-Schmidt, least squares, SVD)
with an interactive visual breakdown of each stage. PSNR improved
from 24 dB (least squares) to 31 dB (truncated SVD at k=20).
