import React from "react";
import { Sparkles, UploadCloud, BarChart3, Sliders, ShieldAlert, Cpu } from "lucide-react";

export default function Sidebar({
  activeTab,
  setActiveTab,
  selectedModel,
  setSelectedModel,
  threshold,
  setThreshold
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <Cpu size={28} color="#7971ea" />
        <span>Churn AI SaaS</span>
      </div>

      <div className="sidebar-section-title">Navigation</div>
      
      <div
        className={`nav-item ${activeTab === "single" ? "active" : ""}`}
        onClick={() => setActiveTab("single")}
      >
        <Sparkles size={18} />
        <span>🔮 Single Predictor</span>
      </div>

      <div
        className={`nav-item ${activeTab === "batch" ? "active" : ""}`}
        onClick={() => setActiveTab("batch")}
      >
        <UploadCloud size={18} />
        <span>📂 Batch & Portfolio Risk</span>
      </div>

      <div
        className={`nav-item ${activeTab === "performance" ? "active" : ""}`}
        onClick={() => setActiveTab("performance")}
      >
        <BarChart3 size={18} />
        <span>📊 Model Performance</span>
      </div>

      <div className="sidebar-section-title">Model Configuration</div>
      <select
        className="sidebar-select"
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
      >
        <option value="ANN">ANN (Neural Network)</option>
        <option value="Random Forest">Random Forest</option>
        <option value="XGBoost">XGBoost Classifier</option>
        <option value="Logistic Regression">Logistic Regression</option>
      </select>

      <div className="sidebar-section-title">Decision Threshold (&Theta;)</div>
      <div className="slider-container">
        <div className="slider-labels">
          <span>Threshold</span>
          <span style={{ fontWeight: "800", color: "#7971ea" }}>{Number(threshold).toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.10"
          max="0.90"
          step="0.05"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="custom-range"
        />
        <div className="threshold-badge">
          {threshold < 0.5 ? "⚡ High-Recall (Aggressive Churn Catch)" : threshold > 0.5 ? "🎯 High-Precision (Strict Churn)" : "⚖️ Balanced (50% Standard)"}
        </div>
      </div>
    </aside>
  );
}
