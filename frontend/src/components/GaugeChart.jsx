import React from "react";

export default function GaugeChart({ probability, threshold = 0.5 }) {
  const probPct = Math.min(100, Math.max(0, probability * 100));
  
  // Radius and arc math for semicircle gauge
  const radius = 80;
  const strokeWidth = 14;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = normalizedRadius * Math.PI;
  const strokeDashoffset = circumference - (probPct / 100) * circumference;

  let gaugeColor = "#10b981"; // green
  if (probPct >= threshold * 100) {
    gaugeColor = "#f43f5e"; // rose/red
  } else if (probPct >= threshold * 60) {
    gaugeColor = "#f59e0b"; // amber
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", position: "relative" }}>
      <svg height="120" width="220" viewBox="0 0 220 120" style={{ overflow: "visible" }}>
        {/* Background Arc */}
        <path
          d="M 20 100 A 80 80 0 0 1 200 100"
          fill="none"
          stroke="rgba(44, 62, 80, 0.08)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        {/* Value Arc */}
        <path
          d="M 20 100 A 80 80 0 0 1 200 100"
          fill="none"
          stroke={gaugeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275), stroke 0.4s ease" }}
        />
      </svg>
      <div style={{ marginTop: "-20px", textAlign: "center" }}>
        <span style={{ fontSize: "2rem", fontWeight: "900", color: "#2C3E50" }}>
          {probPct.toFixed(1)}%
        </span>
        <div style={{ fontSize: "0.8rem", color: "#64748b", fontWeight: "600" }}>
          Churn Probability
        </div>
      </div>
    </div>
  );
}
