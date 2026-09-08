import React, { useState, useEffect } from "react";
import { fetchModelComparison, fetchFeatureImportance, fetchEdaDemographics } from "../api";
import { BarChart3, Award, Zap, Globe, PieChart, Layers } from "lucide-react";

export default function ModelPerformance() {
  const [comparison, setComparison] = useState(null);
  const [importance, setImportance] = useState(null);
  const [demographics, setDemographics] = useState(null);
  const [activeTab, setActiveTab] = useState("comparison");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [compRes, impRes, demoRes] = await Promise.all([
          fetchModelComparison().catch(() => null),
          fetchFeatureImportance().catch(() => null),
          fetchEdaDemographics().catch(() => null)
        ]);
        setComparison(compRes);
        setImportance(impRes);
        setDemographics(demoRes);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div>
      <h1 className="main-header">📊 SYSTEM BENCHMARKS & EXPLAINABILITY</h1>
      <p className="sub-header">
        Multi-model evaluation benchmarks, global feature importance rankings, and customer demographics insights.
      </p>

      {/* Sub-tabs */}
      <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1.5rem" }}>
        <button
          className={activeTab === "comparison" ? "btn-primary" : "btn-secondary"}
          onClick={() => setActiveTab("comparison")}
          style={{ width: "auto" }}
        >
          <Award size={16} />
          <span>Model Comparison Benchmarks</span>
        </button>

        <button
          className={activeTab === "importance" ? "btn-primary" : "btn-secondary"}
          onClick={() => setActiveTab("importance")}
          style={{ width: "auto" }}
        >
          <Zap size={16} />
          <span>Global Feature Importance</span>
        </button>

        <button
          className={activeTab === "demographics" ? "btn-primary" : "btn-secondary"}
          onClick={() => setActiveTab("demographics")}
          style={{ width: "auto" }}
        >
          <Globe size={16} />
          <span>Customer Insights & Demographics</span>
        </button>
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-muted)" }}>
          Loading benchmarks and models analytics...
        </div>
      ) : (
        <>
          {/* Tab 1: Comparison */}
          {activeTab === "comparison" && comparison && (
            <div className="glass-card">
              <div className="card-title">🏆 Classification Performance Metrics Across ML Models</div>

              <div className="custom-table-wrapper" style={{ marginBottom: "1.5rem" }}>
                <table className="custom-table">
                  <thead>
                    <tr>
                      <th>Classifier</th>
                      <th>Accuracy</th>
                      <th>Precision</th>
                      <th>Recall (Key for Churn)</th>
                      <th>F1-Score</th>
                      <th>ROC-AUC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(comparison).map(([name, metrics], idx) => (
                      <tr key={idx}>
                        <td style={{ fontWeight: "800", color: "#2C3E50" }}>{name}</td>
                        <td>{(metrics.Accuracy * 100).toFixed(1)}%</td>
                        <td>{(metrics.Precision * 100).toFixed(1)}%</td>
                        <td style={{ fontWeight: "800", color: "#7971ea" }}>{(metrics.Recall * 100).toFixed(1)}%</td>
                        <td>{(metrics["F1-Score"] * 100).toFixed(1)}%</td>
                        <td>{(metrics["ROC-AUC"] * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{ padding: "1rem", borderRadius: "0.85rem", background: "rgba(121, 113, 234, 0.1)", border: "1px solid rgba(121, 113, 234, 0.25)", color: "#2C3E50", fontSize: "0.92rem", lineHeight: "1.5" }}>
                💡 <strong>Banking Strategic Context:</strong> In bank customer retention, <strong>Recall</strong> is prioritized over raw accuracy because identifying an at-risk customer allows early relationship manager outreach, whereas missing a churning customer results in a permanent loss of account balances.
              </div>
            </div>
          )}

          {/* Tab 2: Feature Importance */}
          {activeTab === "importance" && importance && (
            <div className="grid-2">
              <div className="glass-card">
                <div className="card-title">🌲 Random Forest Feature Importance (MDI)</div>
                <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
                  Relative contribution of each feature in decision tree splits:
                </p>
                {importance["Random Forest"]?.slice(0, 7).map((f, i) => (
                  <div key={i} style={{ marginBottom: "0.75rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.88rem", fontWeight: "700", marginBottom: "0.25rem" }}>
                      <span>{f.Feature}</span>
                      <span style={{ color: "var(--primary-accent)" }}>{(f.Importance * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: "8px", borderRadius: "4px", background: "rgba(44, 62, 80, 0.08)", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${f.Importance * 100 * 2.5}%`,
                          background: "linear-gradient(90deg, #B5FFFC, #7971ea, #ff758c)",
                          borderRadius: "4px"
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="glass-card">
                <div className="card-title">⚡ XGBoost Feature Importance (Gain)</div>
                <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
                  Relative statistical gain per feature during gradient boosting:
                </p>
                {importance["XGBoost"]?.slice(0, 7).map((f, i) => (
                  <div key={i} style={{ marginBottom: "0.75rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.88rem", fontWeight: "700", marginBottom: "0.25rem" }}>
                      <span>{f.Feature}</span>
                      <span style={{ color: "#ff758c" }}>{(f.Importance * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: "8px", borderRadius: "4px", background: "rgba(44, 62, 80, 0.08)", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${f.Importance * 100 * 2.5}%`,
                          background: "linear-gradient(90deg, #B5FFFC, #ff7eb3, #ff758c)",
                          borderRadius: "4px"
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tab 3: Demographics */}
          {activeTab === "demographics" && demographics && (
            <div className="grid-2">
              <div className="glass-card">
                <div className="card-title">🌍 Churn Rate by Geography</div>
                {Object.entries(demographics.geo_churn || {}).map(([country, rate], idx) => (
                  <div key={idx} style={{ marginBottom: "1rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "700", marginBottom: "0.25rem" }}>
                      <span>{country}</span>
                      <span style={{ color: rate > 30 ? "#f43f5e" : "#10b981" }}>{rate}%</span>
                    </div>
                    <div style={{ height: "10px", borderRadius: "5px", background: "rgba(44, 62, 80, 0.08)", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${rate * 2}%`,
                          background: rate > 30 ? "linear-gradient(90deg, #ff7eb3, #f43f5e)" : "linear-gradient(90deg, #B5FFFC, #10b981)",
                          borderRadius: "5px"
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="glass-card">
                <div className="card-title">🛡️ Churn Rate by Number of Products</div>
                {Object.entries(demographics.prod_churn || {}).map(([prod, rate], idx) => (
                  <div key={idx} style={{ marginBottom: "1rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "700", marginBottom: "0.25rem" }}>
                      <span>{prod} Product(s)</span>
                      <span style={{ color: rate > 50 ? "#f43f5e" : rate < 15 ? "#10b981" : "#f59e0b" }}>{rate}%</span>
                    </div>
                    <div style={{ height: "10px", borderRadius: "5px", background: "rgba(44, 62, 80, 0.08)", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${rate}%`,
                          background: rate > 50 ? "linear-gradient(90deg, #ff7eb3, #f43f5e)" : rate < 15 ? "linear-gradient(90deg, #B5FFFC, #10b981)" : "linear-gradient(90deg, #B5FFFC, #f59e0b)",
                          borderRadius: "5px"
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
