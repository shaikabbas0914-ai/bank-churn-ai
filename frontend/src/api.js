const API_BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "") + "/api";

export async function fetchPredictSingle(customerData, modelName = "ANN", threshold = 0.50) {
  const res = await fetch(`${API_BASE}/predict/single`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...customerData,
      model_name: modelName,
      threshold: threshold
    })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Single prediction failed");
  }
  return res.json();
}

export async function fetchWhatIfSimulation(customerData, adjustments, modelName = "ANN", threshold = 0.50) {
  const res = await fetch(`${API_BASE}/predict/what-if`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      customer: customerData,
      adjustments: adjustments,
      model_name: modelName,
      threshold: threshold
    })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "What-If simulation failed");
  }
  return res.json();
}

export async function fetchBatchPredict(file, modelName = "ANN", threshold = 0.50) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_name", modelName);
  formData.append("threshold", threshold);

  const res = await fetch(`${API_BASE}/predict/batch`, {
    method: "POST",
    body: formData
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Batch prediction failed");
  }
  return res.json();
}

export async function fetchModelComparison() {
  const res = await fetch(`${API_BASE}/models/comparison`);
  if (!res.ok) throw new Error("Could not load model comparisons");
  return res.json();
}

export async function fetchFeatureImportance() {
  const res = await fetch(`${API_BASE}/models/feature-importance`);
  if (!res.ok) throw new Error("Could not load feature importances");
  return res.json();
}

export async function fetchEdaDemographics() {
  const res = await fetch(`${API_BASE}/eda/demographics`);
  if (!res.ok) throw new Error("Could not load EDA demographics");
  return res.json();
}

export function getSampleCsvDownloadUrl() {
  return `${API_BASE}/data/sample-csv?n=50`;
}
