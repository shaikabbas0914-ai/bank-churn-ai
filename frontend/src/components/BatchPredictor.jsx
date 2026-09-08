import React, { useState } from "react";
import { fetchBatchPredict, getSampleCsvDownloadUrl } from "../api";
import { UploadCloud, Download, FileText, Filter, CheckCircle2, DollarSign, TrendingDown, Users } from "lucide-react";

export default function BatchPredictor({ selectedModel, threshold }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [filterRisk, setFilterRisk] = useState("All");

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleProcessBatch = async (fileToUpload = file) => {
    if (!fileToUpload) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetchBatchPredict(fileToUpload, selectedModel, threshold);
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const sampleRes = await fetch(getSampleCsvDownloadUrl());
      const blob = await sampleRes.blob();
      const sampleFile = new File([blob], "sample_bank_churn_batch.csv", { type: "text/csv" });
      setFile(sampleFile);
      await handleProcessBatch(sampleFile);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExportCsv = () => {
    if (!data?.records) return;
    const headers = Object.keys(data.records[0]).join(",");
    const rows = data.records.map((r) => Object.values(r).join(",")).join("\n");
    const csvContent = `${headers}\n${rows}`;
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "churn_predictions_with_clv_output.csv";
    link.click();
  };

  const filteredRecords = data?.records?.filter((r) => {
    if (filterRisk === "All") return true;
    return r["Risk Category"] === filterRisk;
  }) || [];

  return (
    <div>
      <h1 className="main-header">📂 BATCH PREDICTIONS & REVENUE RISK</h1>
      <p className="sub-header">
        Bulk customer risk evaluation, portfolio financial valuation, and automated test data generator.
      </p>

      {/* Upload Zone */}
      <div className="glass-card">
        <div className="card-title">Upload Customer Dataset or Use Sample Data</div>
        
        <div className="grid-2">
          <div>
            <div
              style={{
                border: "2px dashed rgba(121, 113, 234, 0.4)",
                borderRadius: "1rem",
                padding: "2rem",
                textAlign: "center",
                background: "rgba(255, 255, 255, 0.6)",
                cursor: "pointer"
              }}
              onClick={() => document.getElementById("csv-file-input").click()}
            >
              <UploadCloud size={40} color="#7971ea" style={{ margin: "0 auto 0.75rem" }} />
              <div style={{ fontWeight: "700", fontSize: "1.05rem" }}>
                {file ? file.name : "Click or drag CSV file here to upload"}
              </div>
              <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                Required columns: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
              </div>
              <input
                id="csv-file-input"
                type="file"
                accept=".csv"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
            </div>

            {file && (
              <div style={{ marginTop: "1rem" }}>
                <button className="btn-primary" onClick={() => handleProcessBatch()} disabled={loading}>
                  <UploadCloud size={18} />
                  <span>{loading ? "Processing Batch..." : "Process Batch Predictions"}</span>
                </button>
              </div>
            )}
          </div>

          <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", gap: "1rem", padding: "1rem" }}>
            <div style={{ fontWeight: "700", fontSize: "1.1rem", color: "#2C3E50" }}>
              Quick Test Batch Dataset
            </div>
            <p style={{ fontSize: "0.9rem", color: "var(--text-muted)" }}>
              Don't have a dataset ready? Generate 50 realistic test bank customer records with one click:
            </p>
            <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
              <a href={getSampleCsvDownloadUrl()} className="btn-secondary" style={{ textDecoration: "none" }}>
                <Download size={16} />
                <span>Download Sample CSV</span>
              </a>
              <button className="btn-primary" onClick={handleLoadSample} disabled={loading} style={{ width: "auto" }}>
                <CheckCircle2 size={16} />
                <span>Load Sample Data Directly</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div style={{ padding: "1rem", borderRadius: "0.75rem", background: "rgba(244, 63, 94, 0.15)", color: "#9f1239", marginBottom: "1.5rem", fontWeight: "600" }}>
          ⚠️ {error}
        </div>
      )}

      {/* Batch Results Overview */}
      {data && (
        <div>
          <div className="card-title" style={{ marginTop: "1rem" }}>
            💰 Portfolio Financial & Churn Risk Metrics
          </div>

          <div className="grid-4" style={{ marginBottom: "1.5rem" }}>
            <div className="kpi-card">
              <div className="kpi-label">Total Customers</div>
              <div className="kpi-value">{data.summary.total_customers}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">High Risk Customers</div>
              <div className="kpi-value" style={{ color: "#f43f5e" }}>
                {data.summary.high_risk_count} ({data.summary.churn_rate_pct}%)
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Total Balance at Risk</div>
              <div className="kpi-value" style={{ color: "#f59e0b" }}>
                ${data.summary.total_balance_at_risk.toLocaleString()}
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Annual Revenue at Risk</div>
              <div className="kpi-value" style={{ color: "#7971ea" }}>
                ${data.summary.total_revenue_at_risk.toLocaleString()}
              </div>
            </div>
          </div>

          {/* Table Container */}
          <div className="glass-card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem", flexWrap: "wrap", gap: "1rem" }}>
              <div className="card-title" style={{ margin: 0, border: "none" }}>
                📋 Enriched Predictions Table ({filteredRecords.length} Customers)
              </div>
              
              <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <Filter size={16} color="var(--text-muted)" />
                  <select
                    className="form-control"
                    value={filterRisk}
                    onChange={(e) => setFilterRisk(e.target.value)}
                    style={{ padding: "0.4rem 0.8rem", fontSize: "0.85rem" }}
                  >
                    <option value="All">All Risk Levels</option>
                    <option value="High Risk">High Risk Only</option>
                    <option value="Medium Risk">Medium Risk Only</option>
                    <option value="Low Risk">Low Risk Only</option>
                  </select>
                </div>

                <button className="btn-secondary" onClick={handleExportCsv} style={{ padding: "0.5rem 1rem", fontSize: "0.85rem" }}>
                  <Download size={14} />
                  <span>Export CSV</span>
                </button>
              </div>
            </div>

            <div className="custom-table-wrapper">
              <table className="custom-table">
                <thead>
                  <tr>
                    <th>Customer</th>
                    <th>Geo / Gender</th>
                    <th>Age</th>
                    <th>Balance ($)</th>
                    <th>Products</th>
                    <th>Churn Prob</th>
                    <th>Risk Category</th>
                    <th>Est. CLV ($)</th>
                    <th>Revenue at Risk ($)</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRecords.slice(0, 30).map((row, idx) => {
                    const isHigh = row["Risk Category"] === "High Risk";
                    const isMed = row["Risk Category"] === "Medium Risk";
                    return (
                      <tr key={idx}>
                        <td>
                          <strong>{row.Surname || `ID: ${row.CustomerId || idx + 1}`}</strong>
                        </td>
                        <td>{row.Geography} ({row.Gender})</td>
                        <td>{row.Age} yrs</td>
                        <td>${Number(row.Balance).toLocaleString()}</td>
                        <td>{row.NumOfProducts}</td>
                        <td style={{ fontWeight: "800", color: isHigh ? "#f43f5e" : isMed ? "#f59e0b" : "#10b981" }}>
                          {row["Churn Probability (%)"]}%
                        </td>
                        <td>
                          <span
                            style={{
                              padding: "0.3rem 0.6rem",
                              borderRadius: "0.5rem",
                              fontSize: "0.78rem",
                              fontWeight: "800",
                              background: isHigh
                                ? "rgba(244, 63, 94, 0.15)"
                                : isMed
                                ? "rgba(245, 158, 11, 0.15)"
                                : "rgba(16, 185, 129, 0.15)",
                              color: isHigh ? "#9f1239" : isMed ? "#92400e" : "#065f46"
                            }}
                          >
                            {row["Risk Category"]}
                          </span>
                        </td>
                        <td>${Number(row["Estimated CLV ($)"]).toLocaleString()}</td>
                        <td style={{ fontWeight: "700", color: isHigh ? "#f43f5e" : "#2C3E50" }}>
                          ${Number(row["Revenue At Risk ($)"]).toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
