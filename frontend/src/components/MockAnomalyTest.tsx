import React from 'react';
import { mockAnomalyExplanations, getMockAnomalyById } from '../data/mockAnomalyExplanations';

const MockAnomalyTest: React.FC = () => {
  const handleShowAnomaly = (id: string) => {
    const mockData = getMockAnomalyById(id);
    if (mockData) {
      console.log('=== MOCK ANOMALY DATA ===');
      console.log('Anomaly:', mockData.anomaly);
      console.log('Explanation:', mockData.explanation);
      alert(`Anomaly ${id}: ${mockData.anomaly.attackType?.name} attack detected!`);
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Mock Anomaly Explanations Test</h2>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        {mockAnomalyExplanations.map((item, index) => (
          <button
            key={index}
            onClick={() => handleShowAnomaly(item.anomaly.id)}
            className="p-2 border rounded hover:bg-gray-100"
          >
            <div className="text-sm font-medium">{item.anomaly.id}</div>
            <div className="text-xs">{item.anomaly.attackType?.name}</div>
            <div className="text-xs text-gray-500">{item.anomaly.severity}</div>
          </button>
        ))}
      </div>
      
      <div className="mt-4 p-4 bg-gray-50 rounded">
        <h3 className="font-medium mb-2">Instructions:</h3>
        <ol className="text-sm list-decimal list-inside space-y-1">
          <li>Click any anomaly button above to see its data in console</li>
          <li>Open browser console (F12) to view detailed anomaly and explanation data</li>
          <li>Each anomaly has different attack types: Botnet, DoS, Infiltration, Other, PortScan</li>
          <li>The data includes network details, feature importance, and recommendations</li>
        </ol>
      </div>
    </div>
  );
};

export default MockAnomalyTest;
