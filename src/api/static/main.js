async function handleSubmit(event) {
  event.preventDefault();

  const text = document.getElementById("newsInput").value.trim();
  const resultDiv = document.getElementById("result");
  const btn = document.getElementById("submitBtn");

  if (text.length < 100) {
    resultDiv.classList.remove("hidden");
    resultDiv.innerHTML = `
      <div class="result-tag fake">Texto muito curto</div>
      <p style="margin-top: 0.5rem;">Insira ao menos 100 caracteres.</p>
    `;
    return;
  }

  btn.classList.add("loading");
  btn.disabled = true;
  resultDiv.classList.add("hidden");

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    const isReal = data.prediction === "REAL";
    const confidence = parseFloat(data.confidence.replace("%", ""));

    resultDiv.classList.remove("hidden");
    resultDiv.innerHTML = `
      <div class="result-tag ${isReal ? "real" : "fake"}">${
      data.prediction
    }</div>

      <div class="confidence-section">
        <div class="confidence-label">
          <strong>Nível de confiança:</strong> ${data.confidence}
        </div>
        <div class="confidence-bar">
          <div class="confidence-fill" style="width: ${confidence}%"></div>
        </div>
      </div>
    `;
  } catch (err) {
    resultDiv.classList.remove("hidden");
    resultDiv.innerHTML = `
      <div class="result-tag fake">Erro ao conectar</div>
      <p style="margin-top: 0.5rem;">Certifique-se de que a API está ativa.</p>
    `;
  }

  btn.classList.remove("loading");
  btn.disabled = false;
}
