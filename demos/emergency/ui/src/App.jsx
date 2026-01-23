/**
 * App
 *
 * Thin wrapper around the emergency demo UI.
 */

import "./App.css";
import EmergencyVisualization from "./emergency_visualization.jsx";

export default function App() {
  return (
    <div style={{ padding: 16 }}>
      <EmergencyVisualization />
    </div>
  );
}
