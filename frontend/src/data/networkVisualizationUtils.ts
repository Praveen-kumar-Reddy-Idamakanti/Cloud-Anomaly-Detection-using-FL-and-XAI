import { TimeSeriesData } from '../api/api';

// Network activity patterns for different scenarios
export const NETWORK_SCENARIOS = {
  NORMAL: 'normal',
  SPIKE: 'spike',
  GRADUAL_INCREASE: 'gradual_increase',
  CYCLIC: 'cyclic',
  ANOMALY_BURST: 'anomaly_burst'
} as const;

// Generate network activity based on scenario
export const generateScenarioData = (
  scenario: keyof typeof NETWORK_SCENARIOS,
  days: number = 30
): TimeSeriesData[] => {
  const data: TimeSeriesData[] = [];
  const today = new Date();
  
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    
    let logs = 30000;
    let anomalies = 150;
    
    switch (scenario) {
      case 'NORMAL':
        // Normal daily variation
        logs = 30000 + Math.sin(i * 0.3) * 5000 + Math.random() * 3000;
        anomalies = 150 + Math.sin(i * 0.3) * 30 + Math.random() * 20;
        break;
        
      case 'SPIKE':
        // Normal with occasional spikes
        if (i === 15 || i === 7) {
          logs = 80000 + Math.random() * 10000;
          anomalies = 500 + Math.random() * 100;
        } else {
          logs = 30000 + Math.random() * 5000;
          anomalies = 150 + Math.random() * 20;
        }
        break;
        
      case 'GRADUAL_INCREASE':
        // Gradual increase over time
        logs = 20000 + (i * 1000) + Math.random() * 2000;
        anomalies = 100 + (i * 5) + Math.random() * 10;
        break;
        
      case 'CYCLIC':
        // Weekly pattern
        const dayOfWeek = i % 7;
        const weeklyMultiplier = dayOfWeek < 5 ? 1.2 : 0.7; // Weekdays higher
        logs = 30000 * weeklyMultiplier + Math.sin(i * 0.5) * 3000 + Math.random() * 2000;
        anomalies = 150 * weeklyMultiplier + Math.sin(i * 0.5) * 20 + Math.random() * 15;
        break;
        
      case 'ANOMALY_BURST':
        // Normal with anomaly bursts
        logs = 30000 + Math.random() * 5000;
        if (i >= 10 && i <= 15) {
          anomalies = 400 + Math.random() * 100; // Anomaly burst
        } else {
          anomalies = 150 + Math.random() * 20;
        }
        break;
    }
    
    data.push({
      date: date.toISOString().split('T')[0],
      logs: Math.floor(Math.max(1000, logs)),
      anomalies: Math.floor(Math.max(10, anomalies))
    });
  }
  
  return data;
};

// Calculate network statistics
export const calculateNetworkStats = (data: TimeSeriesData[]) => {
  if (data.length === 0) {
    return {
      totalLogs: 0,
      totalAnomalies: 0,
      avgLogs: 0,
      avgAnomalies: 0,
      peakLogs: 0,
      peakAnomalies: 0,
      anomalyRate: 0
    };
  }
  
  const totalLogs = data.reduce((sum, item) => sum + item.logs, 0);
  const totalAnomalies = data.reduce((sum, item) => sum + item.anomalies, 0);
  const avgLogs = totalLogs / data.length;
  const avgAnomalies = totalAnomalies / data.length;
  const peakLogs = Math.max(...data.map(item => item.logs));
  const peakAnomalies = Math.max(...data.map(item => item.anomalies));
  const anomalyRate = totalAnomalies > 0 ? (totalAnomalies / totalLogs) * 100 : 0;
  
  return {
    totalLogs,
    totalAnomalies,
    avgLogs: Math.floor(avgLogs),
    avgAnomalies: Math.floor(avgAnomalies),
    peakLogs,
    peakAnomalies,
    anomalyRate: anomalyRate.toFixed(2)
  };
};

// Detect anomalous patterns in data
export const detectAnomalousPatterns = (data: TimeSeriesData[]) => {
  if (data.length < 7) return [];
  
  const anomalies: Array<{
    date: string;
    type: 'volume_spike' | 'anomaly_spike' | 'unusual_pattern';
    severity: 'low' | 'medium' | 'high';
    description: string;
  }> = [];
  
  // Calculate averages and standard deviations
  const logs = data.map(d => d.logs);
  const anomalyCounts = data.map(d => d.anomalies);
  
  const avgLogs = logs.reduce((a, b) => a + b, 0) / logs.length;
  const avgAnomalies = anomalyCounts.reduce((a, b) => a + b, 0) / anomalyCounts.length;
  
  const logsStdDev = Math.sqrt(logs.reduce((sum, val) => sum + Math.pow(val - avgLogs, 2), 0) / logs.length);
  const anomaliesStdDev = Math.sqrt(anomalyCounts.reduce((sum, val) => sum + Math.pow(val - avgAnomalies, 2), 0) / anomalyCounts.length);
  
  // Detect spikes
  data.forEach((item, index) => {
    if (item.logs > avgLogs + 2 * logsStdDev) {
      anomalies.push({
        date: item.date,
        type: 'volume_spike',
        severity: item.logs > avgLogs + 3 * logsStdDev ? 'high' : 'medium',
        description: `Unusual network traffic spike: ${item.logs.toLocaleString()} logs`
      });
    }
    
    if (item.anomalies > avgAnomalies + 2 * anomaliesStdDev) {
      anomalies.push({
        date: item.date,
        type: 'anomaly_spike',
        severity: item.anomalies > avgAnomalies + 3 * anomaliesStdDev ? 'high' : 'medium',
        description: `High anomaly count: ${item.anomalies} anomalies detected`
      });
    }
  });
  
  return anomalies;
};

// Format data for different chart types
export const formatForChart = {
  // Format for area charts (continuous data)
  area: (data: TimeSeriesData[]) => data.map(item => ({
    ...item,
    displayDate: new Date(item.date).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric' 
    })
  })),
  
  // Format for bar charts (comparative data)
  bar: (data: TimeSeriesData[]) => data.map(item => ({
    ...item,
    displayDate: new Date(item.date).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric' 
    })
  })),
  
  // Format for line charts (trends)
  line: (data: TimeSeriesData[]) => data.map(item => ({
    ...item,
    displayDate: new Date(item.date).toLocaleDateString('en-US', { 
      weekday: 'short' 
    })
  }))
};

// Color schemes for different visualizations
export const CHART_COLORS = {
  primary: '#8884d8',
  secondary: '#ff4557',
  success: '#52c41a',
  warning: '#faad14',
  danger: '#ff4d4f',
  info: '#1890ff',
  
  gradients: {
    logs: {
      start: '#8884d8',
      end: 'rgba(136, 132, 216, 0.1)'
    },
    anomalies: {
      start: '#ff4557',
      end: 'rgba(255, 69, 87, 0.1)'
    }
  }
};

// Export utilities
export default {
  generateScenarioData,
  calculateNetworkStats,
  detectAnomalousPatterns,
  formatForChart,
  CHART_COLORS,
  NETWORK_SCENARIOS
};
