import React, { useState, useEffect } from "react";
import { fetchPredictSingle, fetchWhatIfSimulation } from "../api";
import GaugeChart from "./GaugeChart";
import { Sparkles, Download, CheckCircle, AlertTriangle, XCircle, ArrowDownRight, RefreshCw, Zap } from "lucide-react";
import confetti from "canvas-confetti";

export default function SinglePredictor({ selectedModel, threshold }) {
  const [formData, setFormData] = useState({
    CreditScore: 650,
    Geography: "France",
    Gender: "Female",
    Age: 42,
    Tenure: 4,
    Balance: 85000.0,
    NumOfProducts: 1,
    HasCrCard: 1,
    IsActiveMember: 0,
    EstimatedSalary: 95000.0
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // What-If State
  const [simAdjustments, setSimAdjustments] = useState({
    IsActiveMember: 1,
    NumOfProducts: 2,
    HasCrCard: 1
  });
  const [simResult, setSimResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);

  // Run prediction
  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPredictSingle(formData, selectedModel, threshold);
      setResult(data);
      // Also run default simulation
      handleSimulate(data.prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Run What-If Simulation
  const handleSimulate = async (basePred) => {
    setSimLoading(true);
    try {
      const data = await fetchWhatIfSimulation(formData, simAdjustments, selectedModel, threshold);
      setSimResult(data);
      // Trigger confetti if retention reduces churn by > 20%
      if (data.delta_pct <= -20) {
        confetti({
          particleCount: 50,
          spread: 60,
          origin: { y: 0.8 },
          colors: ['#ff758c', '#7971ea', '#06b6d4', '#10b981']
        });
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSimLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    handlePredict();
  }, [selectedModel, threshold]);

  const handleInputChange = (field, val) => {
    setFormData((prev) => ({ ...prev, [field]: val }));
  };

  const handleDownloadReport = () => {
    if (!result?.report_text) return;
    const blob = new Blob([result.report_text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `churn_memo_customer_${formData.CreditScore}_${formData.Age}.txt`;
    link.click();
  };

  return (
    <div>
      <h1 className="main-header">🏦 BANK CUSTOMER CHURN AI</h1>
      <p className="sub-header">
        Real-time risk scoring, Explainable AI factor attribution, CLV estimation & What-If retention simulation.
      </p>

      {error && (
        <div style={{ padding: "1rem", borderRadius: "0.75rem", background: "rgba(244, 63, 94, 0.15)", color: "#9f1239", marginBottom: "1.5rem", fontWeight: "600" }}>
          ⚠️ {error}
        </div>
      )}

      <div className="grid-2">
        {/* Input Form Card */}
        <div className="glass-card">
          <div className="card-title">Customer Demographics & Account Details</div>
          
          <div className="grid-2" style={{ gap: "1rem" }}>
            <div className="form-group">
              <label className="form-label">
                <span>Credit Score</span>
                <span style={{ color: "var(--primary-accent)", fontWeight: "800" }}>{formData.CreditScore}</span>
              </label>
              <input
                type="range"
                min="300"
                max="850"
                value={formData.CreditScore}
                onChange={(e) => handleInputChange("CreditScore", parseInt(e.target.value))}
                className="custom-range"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Geography / Country</label>
              <select
                className="form-control"
                value={formData.Geography}
                onChange={(e) => handleInputChange("Geography", e.target.value)}
              >
                <option value="India">🇮🇳 India</option>
                <option value="United States">🇺🇸 United States</option>
                <option value="United Kingdom">🇬🇧 United Kingdom</option>
                <option value="Canada">🇨🇦 Canada</option>
                <option value="Australia">🇦🇺 Australia</option>
                <option value="France">🇫🇷 France</option>
                <option value="Germany">🇩🇪 Germany</option>
                <option value="Spain">🇪🇸 Spain</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Gender</label>
              <select
                className="form-control"
                value={formData.Gender}
                onChange={(e) => handleInputChange("Gender", e.target.value)}
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">
                <span>Age</span>
                <span style={{ color: "var(--primary-accent)", fontWeight: "800" }}>{formData.Age} yrs</span>
              </label>
              <input
                type="range"
                min="18"
                max="90"
                value={formData.Age}
                onChange={(e) => handleInputChange("Age", parseInt(e.target.value))}
                className="custom-range"
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                <span>Tenure (Years)</span>
                <span style={{ color: "var(--primary-accent)", fontWeight: "800" }}>{formData.Tenure} yrs</span>
              </label>
              <input
                type="range"
                min="0"
                max="10"
                value={formData.Tenure}
                onChange={(e) => handleInputChange("Tenure", parseInt(e.target.value))}
                className="custom-range"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Account Balance ($)</label>
              <input
                type="number"
                step="1000"
                value={formData.Balance}
                onChange={(e) => handleInputChange("Balance", parseFloat(e.target.value) || 0)}
                className="form-control"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Number of Products</label>
              <select
                className="form-control"
                value={formData.NumOfProducts}
                onChange={(e) => handleInputChange("NumOfProducts", parseInt(e.target.value))}
              >
                <option value={1}>1 Product</option>
                <option value={2}>2 Products (Optimal)</option>
                <option value={3}>3 Products</option>
                <option value={4}>4 Products</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Has Credit Card?</label>
              <select
                className="form-control"
                value={formData.HasCrCard}
                onChange={(e) => handleInputChange("HasCrCard", parseInt(e.target.value))}
              >
                <option value={1}>Yes (Active Card)</option>
                <option value={0}>No</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Active Member Status</label>
              <select
                className="form-control"
                value={formData.IsActiveMember}
                onChange={(e) => handleInputChange("IsActiveMember", parseInt(e.target.value))}
              >
                <option value={1}>Active Member</option>
                <option value={0}>Inactive Member</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Estimated Salary ($)</label>
              <input
                type="number"
                step="5000"
                value={formData.EstimatedSalary}
                onChange={(e) => handleInputChange("EstimatedSalary", parseFloat(e.target.value) || 0)}
                className="form-control"
              />
            </div>
          </div>

          <div style={{ marginTop: "1.5rem" }}>
            <button className="btn-primary" onClick={handlePredict} disabled={loading}>
              <Sparkles size={18} />
              <span>{loading ? "Calculating..." : "Calculate Churn Intelligence"}</span>
            </button>
          </div>
        </div>

        {/* Output Assessment Card */}
        <div className="glass-card">
          <div className="card-title">AI Assessment & Risk Valuation</div>

          {result ? (
            <div>
              {/* Risk Banner */}
              <div
                className={`risk-banner ${
                  result.prediction.risk_category === "Low Risk"
                    ? "risk-low"
                    : result.prediction.risk_category === "Medium Risk"
                    ? "risk-medium"
                    : "risk-high"
                }`}
              >
                {result.prediction.risk_category.toUpperCase()} ({(result.prediction.probability * 100).toFixed(1)}%)
              </div>

              {/* Radial Gauge */}
              <GaugeChart probability={result.prediction.probability} threshold={threshold} />

              {/* KPI Cards Row */}
              <div className="grid-2" style={{ marginTop: "1.25rem" }}>
                <div className="kpi-card">
                  <div className="kpi-label">Est. Lifetime Value (CLV)</div>
                  <div className="kpi-value">${result.clv.clv.toLocaleString()}</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Annual Bank Revenue</div>
                  <div className="kpi-value">${result.clv.annual_revenue.toLocaleString()}</div>
                </div>
              </div>

              {/* Strategic Recommendation */}
              <div className="recommendation-box">
                <strong>Strategic Action: </strong>
                {result.prediction.recommendation}
              </div>
            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "2rem", color: "var(--text-muted)" }}>
              Loading assessment...
            </div>
          )}
        </div>
      </div>

      {/* Explainable AI & What-If Simulator Row */}
      {result && (
        <div className="grid-2" style={{ marginTop: "1.5rem" }}>
          {/* Explainable AI */}
          <div className="glass-card">
            <div className="card-title">🔍 Explainable AI: Risk Drivers & Protective Factors</div>
            <p style={{ fontSize: "0.88rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
              Attribution analysis of customer traits influencing the churn scoring:
            </p>

            {result.explanations?.map((exp, idx) => {
              const isDanger = exp.impact === "Increases Risk";
              return (
                <div key={idx} className={`factor-pill ${isDanger ? "factor-positive" : "factor-negative"}`}>
                  <div>
                    <strong>{isDanger ? "⚠️" : "🛡️"} {exp.feature}</strong> ({exp.value}): {exp.description}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Interactive What-If Retention Simulator */}
          <div className="glass-card">
            <div className="card-title">💡 Interactive 'What-If' Retention Simulator</div>
            <p style={{ fontSize: "0.88rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
              Simulate customer intervention actions in real time to calculate risk reduction (&Delta;%):
            </p>

            <div className="grid-3" style={{ gap: "0.75rem", marginBottom: "1rem" }}>
              <div className="form-group">
                <label className="form-label" style={{ fontSize: "0.8rem" }}>Simulate Member</label>
                <select
                  className="form-control"
                  value={simAdjustments.IsActiveMember}
                  onChange={(e) => {
                    const next = { ...simAdjustments, IsActiveMember: parseInt(e.target.value) };
                    setSimAdjustments(next);
                  }}
                >
                  <option value={1}>Active Member (1)</option>
                  <option value={0}>Inactive Member (0)</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" style={{ fontSize: "0.8rem" }}>Simulate Products</label>
                <select
                  className="form-control"
                  value={simAdjustments.NumOfProducts}
                  onChange={(e) => {
                    const next = { ...simAdjustments, NumOfProducts: parseInt(e.target.value) };
                    setSimAdjustments(next);
                  }}
                >
                  <option value={1}>1 Product</option>
                  <option value={2}>2 Products</option>
                  <option value={3}>3 Products</option>
                  <option value={4}>4 Products</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" style={{ fontSize: "0.8rem" }}>Simulate Card</label>
                <select
                  className="form-control"
                  value={simAdjustments.HasCrCard}
                  onChange={(e) => {
                    const next = { ...simAdjustments, HasCrCard: parseInt(e.target.value) };
                    setSimAdjustments(next);
                  }}
                >
                  <option value={1}>Has Card (1)</option>
                  <option value={0}>No Card (0)</option>
                </select>
              </div>
            </div>

            <div style={{ marginBottom: "1rem" }}>
              <button className="btn-secondary" onClick={() => handleSimulate()} style={{ width: "100%" }}>
                <RefreshCw size={16} />
                <span>Simulate Retention Scenario</span>
              </button>
            </div>

            {simResult && (
              <div>
                <div className="grid-3" style={{ gap: "0.75rem", marginBottom: "1rem" }}>
                  <div className="kpi-card" style={{ padding: "0.75rem" }}>
                    <div className="kpi-label">Current Risk</div>
                    <div className="kpi-value" style={{ fontSize: "1.4rem", color: "#f43f5e" }}>
                      {(simResult.base_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="kpi-card" style={{ padding: "0.75rem" }}>
                    <div className="kpi-label">Simulated Risk</div>
                    <div className="kpi-value" style={{ fontSize: "1.4rem", color: "#10b981" }}>
                      {(simResult.simulated_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="kpi-card" style={{ padding: "0.75rem" }}>
                    <div className="kpi-label">Risk Change (&Delta;)</div>
                    <div className="kpi-value" style={{ fontSize: "1.4rem", color: simResult.delta_pct <= 0 ? "#10b981" : "#f43f5e" }}>
                      {simResult.delta_pct > 0 ? "+" : ""}{simResult.delta_pct.toFixed(1)}%
                    </div>
                  </div>
                </div>

                {simResult.delta_prob < 0 ? (
                  <div style={{ padding: "0.85rem", borderRadius: "0.75rem", background: "rgba(16, 185, 129, 0.12)", border: "1px solid rgba(16, 185, 129, 0.3)", color: "#065f46", fontSize: "0.9rem", fontWeight: "600" }}>
                    🎉 <strong>Opportunity:</strong> This action reduces churn probability by {Math.abs(simResult.delta_pct).toFixed(1)}% (moving from {simResult.base_risk} to {simResult.simulated_risk}).
                  </div>
                ) : (
                  <div style={{ padding: "0.85rem", borderRadius: "0.75rem", background: "rgba(244, 63, 94, 0.12)", border: "1px solid rgba(244, 63, 94, 0.3)", color: "#9f1239", fontSize: "0.9rem", fontWeight: "600" }}>
                    ⚠️ Scenario projects a {simResult.delta_pct.toFixed(1)}% increase in churn propensity.
                  </div>
                )}
              </div>
            )}

            <div style={{ marginTop: "1.25rem" }}>
              <button className="btn-secondary" onClick={handleDownloadReport} style={{ width: "100%" }}>
                <Download size={16} />
                <span>Download Executive Churn Memo (.TXT)</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
