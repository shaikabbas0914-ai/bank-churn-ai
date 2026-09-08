import React, { useState } from "react";
import Sidebar from "./components/Sidebar";
import SinglePredictor from "./components/SinglePredictor";
import BatchPredictor from "./components/BatchPredictor";
import ModelPerformance from "./components/ModelPerformance";

export default function App() {
  const [activeTab, setActiveTab] = useState("single");
  const [selectedModel, setSelectedModel] = useState("ANN");
  const [threshold, setThreshold] = useState(0.50);

  return (
    <div className="app-container">
      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
        threshold={threshold}
        setThreshold={setThreshold}
      />

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === "single" && (
          <SinglePredictor selectedModel={selectedModel} threshold={threshold} />
        )}
        {activeTab === "batch" && (
          <BatchPredictor selectedModel={selectedModel} threshold={threshold} />
        )}
        {activeTab === "performance" && <ModelPerformance />}
      </main>
    </div>
  );
}
