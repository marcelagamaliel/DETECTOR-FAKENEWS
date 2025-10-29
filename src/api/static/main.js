async function handleSubmit(event) {
  event.preventDefault();

  const text = document.getElementById("newsInput").value;
  const resultDiv = document.getElementById("result");

  resultDiv.innerHTML = "🔍 Analisando...";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    resultDiv.innerHTML = `
      <p><strong>Resultado:</strong> ${data.prediction}</p>
      <p><em>Confiança:</em> ${data.confidence}</p>
    `;
  } catch (error) {
    resultDiv.innerHTML = "❌ Erro ao conectar à API.";
  }
}
