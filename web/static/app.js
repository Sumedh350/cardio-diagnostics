async function predict() {
  const raw = document.getElementById("features").value;
  const features = raw.split(",").map(Number);
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  document.getElementById("result").textContent = JSON.stringify(await res.json(), null, 2);
}

async function checkHealth() {
  const res = await fetch("/health");
  document.getElementById("health").textContent = JSON.stringify(await res.json(), null, 2);
}
