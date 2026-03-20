const input = document.getElementById("img-input");
const btn = document.getElementById("run-btn");
const uploadText = document.getElementById("upload-text");
const pipeline = document.getElementById("pipeline");

const STEPS_META = [
  { title: "Image → pixel matrix", subtitle: "Matrix representation · linear transformations", badge: "Real-world data" },
  { title: "Blur matrix + RREF", subtitle: "Gaussian elimination · LU decomposition", badge: "Matrix simplification" },
  { title: "Subspaces", subtitle: "Column space · null space · rank-nullity", badge: "Structure of the space" },
  { title: "Linear independence + basis", subtitle: "Independent columns · basis selection", badge: "Remove redundancy" },
  { title: "Gram-Schmidt", subtitle: "Orthogonal basis from blur columns", badge: "Orthogonalization" },
  { title: "Projection", subtitle: "Project blurred image onto column space", badge: "Projection" },
  { title: "Least squares recovery", subtitle: "x̂ = (AᵀA)⁻¹Aᵀb", badge: "Prediction" },
  { title: "Eigenvalue spectrum", subtitle: "Dominant trends in KᵀK", badge: "Pattern discovery" },
  { title: "SVD reconstruction", subtitle: "Truncated SVD at k = 5, 20, 50", badge: "System simplification" },
];

const INSIGHTS = [
  "Each pixel becomes a number. The full image is represented as a matrix — this is our raw data entering the pipeline.",
  "We construct a blur matrix K and apply it as a matrix multiplication. RREF reveals the pivot structure and confirms rank loss.",
  "Rank tells us how much information survives the blur. Nullity is what's lost — the null space shows unrecoverable directions.",
  "We extract only the linearly independent columns of K to form a basis — removing redundant directions.",
  "Gram-Schmidt converts the basis into orthogonal vectors so each direction is truly independent. Dot product ≈ 0 confirms it.",
  "We project the blurred image onto the column space of K — finding the closest consistent point to our measurement.",
  "Least squares finds the best approximate solution, minimising ‖Kx − b‖². A cleaner result than direct inversion.",
  "Eigenvalues of KᵀK reveal the dominant patterns. A steep drop-off means a few directions carry most of the information.",
  "SVD lets us keep only the top-k singular values. k=20 gives the best balance — enough signal, noise suppressed.",
];

input.addEventListener("change", () => {
  if (input.files[0]) {
    uploadText.textContent = input.files[0].name;
    btn.disabled = false;
  }
});

btn.addEventListener("click", async () => {
  if (!input.files[0]) return;

  pipeline.classList.remove("hidden");
  pipeline.innerHTML = `<div class="loading">Running pipeline...</div>`;
  btn.disabled = true;

  const form = new FormData();
  form.append("image", input.files[0]);

  try {
    const res = await fetch("http://localhost:5000/api/process", {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    renderPipeline(data);
  } catch (err) {
    pipeline.innerHTML = `<div class="loading">Error: could not reach backend. Is Flask running?</div>`;
  }

  btn.disabled = false;
});


function renderPipeline(data) {
  pipeline.innerHTML = "";
  [
    renderStep1(data.step1),
    renderStep2(data.step2),
    renderStep3(data.step3),
    renderStep4(data.step4),
    renderStep5(data.step5),
    renderStep6(data.step6),
    renderStep7(data.step7),
    renderStep8(data.step8),
    renderStep9(data.step9),
  ].forEach(card => pipeline.appendChild(card));
}


function makeCard(index) {
  const m = STEPS_META[index];
  const card = document.createElement("div");
  card.className = "step-card";
  card.innerHTML = `
    <div class="step-header">
      <div class="step-num">${index + 1}</div>
      <div>
        <div class="step-title">${m.title}</div>
        <div class="step-subtitle">${m.subtitle}</div>
      </div>
      <span class="badge">${m.badge}</span>
    </div>
    <div class="card-body"></div>
    <div class="insight">${INSIGHTS[index]}</div>
  `;
  return card;
}

function body(card) { return card.querySelector(".card-body"); }

function imgBlock(b64, label, best = false, extraClass = "") {
  return `<div class="img-block${best ? " best" : ""}${extraClass ? " " + extraClass : ""}">
    <img src="data:image/png;base64,${b64}" />
    <span class="img-lbl${best ? " accent" : ""}">${label}</span>
  </div>`;
}

function matrixBox(rows) {
  return `<div class="matrix-box">${rows.map(r =>
    "[ " + r.map(v => String(v).padStart(7)).join("  ") + " ]"
  ).join("<br>")}<br>[ &nbsp;&nbsp;⋮ &nbsp;&nbsp;&nbsp;⋮ &nbsp;&nbsp;&nbsp;⋮ &nbsp;&nbsp;⋱ ]</div>`;
}

function metrics(items) {
  return `<div class="metric-row">${items.map(([val, lbl]) =>
    `<div class="metric"><div class="metric-val">${val}</div><div class="metric-lbl">${lbl}</div></div>`
  ).join("")}</div>`;
}

function arrow() {
  return `<span class="arrow-sep">→</span>`;
}


function renderStep1(s) {
  const card = makeCard(0);
  body(card).innerHTML = `
    <div class="img-row">
      ${imgBlock(s.original_image, "original image")}
      ${arrow()}
      ${matrixBox(s.matrix_preview)}
    </div>
    ${metrics([
    [s.shape[0] + "×" + s.shape[1], "matrix size"],
    [s.shape[0] * s.shape[1], "total pixels"]
  ])}
  `;
  return card;
}

function renderStep2(s) {
  const card = makeCard(1);
  body(card).innerHTML = `
    <div class="img-row">
      ${imgBlock(s.blurred_image, "blurred image")}
      ${arrow()}
      ${matrixBox(s.rref_preview)}
    </div>
    ${metrics([
    [s.rank, "rank"],
    [s.nullity, "nullity"],
    [s.pivots.length, "pivot columns"]
  ])}
  `;
  return card;
}

function renderStep3(s) {
  const card = makeCard(2);
  body(card).innerHTML = `
    ${metrics([
    [s.rank, "rank"],
    [s.nullity, "nullity"]
  ])}
    <div style="font-size:0.78rem;color:#888;margin-top:8px;">${s.theorem}</div>
  `;
  return card;
}

function renderStep4(s) {
  const card = makeCard(3);
  body(card).innerHTML = `
    ${metrics([
    [s.rank, "independent cols"],
    [s.singular_values_preview[0], "largest singular value"]
  ])}
    <div style="font-size:0.76rem;color:#888;margin-top:10px;">
      top singular values: ${s.singular_values_preview.join(", ")}
    </div>
  `;
  return card;
}

function renderStep5(s) {
  const card = makeCard(4);
  body(card).innerHTML = `
    ${metrics([
    [s.num_vectors_in, "vectors in"],
    [s.num_vectors_out, "orthogonal vectors out"],
    [s.dot_product_check, "dot product (≈ 0?)"]
  ])}
    <div style="font-size:0.78rem;margin-top:8px;color:${s.is_orthogonal ? "#0F6E56" : "#993C1D"};">
      ${s.is_orthogonal ? "✓ orthogonality confirmed" : "⚠ check your vectors"}
    </div>
  `;
  return card;
}

function renderStep6(s) {
  const card = makeCard(5);
  body(card).innerHTML = `
    <div class="img-row">
      ${imgBlock(s.projected_image, "projected image")}
    </div>
    ${metrics([
    [s.is_idempotent ? "yes" : "no", "P² = P (idempotent)?"]
  ])}
  `;
  return card;
}

function renderStep7(s) {
  const card = makeCard(6);
  body(card).innerHTML = `
    <div class="img-row">
      ${imgBlock(s.recovered_image, "least squares recovered")}
    </div>
    ${metrics([
    [s.psnr + " dB", "PSNR"]
  ])}
  `;
  return card;
}

function renderStep8(s) {
  const card = makeCard(7);
  body(card).innerHTML = `
    <div class="img-row">
      <img class="spectrum-plot" src="data:image/png;base64,${s.spectrum_plot}" />
    </div>
    ${metrics([
    [s.dominant_eigenvalue, "dominant eigenvalue"]
  ])}
    <div style="font-size:0.76rem;color:#888;margin-top:8px;">
      top eigenvalues: ${s.eigenvalues_preview.join(", ")}
    </div>
  `;
  return card;
}

function renderStep9(s) {
  const card = makeCard(8);
  const r = s.reconstructions;
  const keys = Object.keys(r);
  const best = keys.reduce((a, b) => r[a].psnr > r[b].psnr ? a : b);
  body(card).innerHTML = `
    <div class="img-row">
      ${keys.map((k, i) => `
        ${i > 0 ? arrow() : ""}
        <div class="img-block svd${k === best ? " best" : ""}">
          <img src="data:image/png;base64,${r[k].image}" />
          <span class="img-lbl${k === best ? " accent" : ""}">
            k = ${k.replace("k", "")} · ${r[k].psnr} dB
          </span>
        </div>
      `).join("")}
    </div>
    ${metrics([
    [r[best].psnr + " dB", "best PSNR"],
    [r[best].ssim, "SSIM"],
    [best.replace("k", ""), "optimal k"]
  ])}
  `;
  return card;
}
