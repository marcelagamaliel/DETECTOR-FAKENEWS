async function handleSubmit(event) {
  event.preventDefault();

  const text = document.getElementById("newsInput").value;
  const resultDiv = document.getElementById("result");

  resultDiv.innerHTML = "🔍 Analisando...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    resultDiv.innerHTML = `
      <strong>Essa noticia parece ser</strong> ${data.prediction}
    `;
  } catch (error) {
    resultDiv.innerHTML = "❌ Erro ao conectar à API.";
  }
}
