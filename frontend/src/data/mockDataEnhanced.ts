// Enhanced mock data for two-stage detection testing
import { AnomalyData, StatData, TimeSeriesData, LogData } from '../api/api';

// Generate random IP addresses
const generateIp = () => {
  return `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`;
};

// Attack types for two-stage detection
const attackTypes = ['Botnet', 'DoS', 'Infiltration', 'Other', 'PortScan'];
const attackTypeColors = ['#10b981', '#ef4444', '#f97316', '#eab308', '#f59e0b'];
const attackSeverities = ['low', 'medium', 'high', 'critical'];

// Generate mock anomalies with attack type information
export const mockAnomaliesEnhanced: AnomalyData[] = 
Array.from({ length: 50 }, (_, i) => {
  const severity = attackSeverities[Math.floor(Math.random() * 4)] as AnomalyData['severity'];
  const sourceIp = generateIp();
  const destinationIp = generateIp();
  const protocol = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'FTP', 'SSH'][Math.floor(Math.random() * 6)] as AnomalyData['protocol'];
  const action = ['blocked', 'allowed', 'flagged', 'quarantined'][Math.floor(Math.random() * 4)] as AnomalyData['action'];
  const confidence = Math.round(Math.random() * 100) / 100;
  const reviewed = Math.random() > 0.7;
  
  // Attack type information for two-stage detection
  const isAnomaly = Math.random() > 0.2;
  const attackTypeIndex = isAnomaly ? Math.floor(Math.random() * 5) : -1; // 0-4 = attack types, -1 = no attack
  
  const attackType = isAnomaly ? {
    id: attackTypeIndex,
    name: attackTypes[attackTypeIndex],
    description: `Attack type: ${attackTypes[attackTypeIndex]}`,
    severity: attackSeverities[Math.min(attackTypeIndex, 3)] as AnomalyData['severity'], // Scale severity
    color: attackTypeColors[attackTypeIndex]
  } : undefined;
  
  const attackConfidence = isAnomaly ? Math.random() * 0.4 + 0.6 : undefined;
  
  return {
    id: `anomaly-${i + 1}`,
    timestamp: new Date(Date.now() - Math.floor(Math.random() * 7 * 24 * 60 * 60 * 1000)).toISOString(),
    severity,
    sourceIp,
    destinationIp,
    protocol,
    action,
    confidence,
    reviewed,
    details: [
      'Unusual traffic pattern detected between hosts',
      'Multiple failed login attempts',
      'Suspicious outbound data transfer',
      'Port scanning activity',
      'Potential data exfiltration',
      'Unusual API access pattern',
      'Credential stuffing attempt',
      'Brute force attack detected',
      'Abnormal data transfer volume'
    ][Math.floor(Math.random() * 9)],
    // Enhanced fields for two-stage detection
    attackType,
    attackConfidence,
    anomalyScore: Math.random() * 0.5,
    reconstructionError: Math.random() * 0.3,
    features: JSON.stringify(Array.from({ length: 78 }, () => Math.random() * 2 - 1))
  };
}).sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

// Generate mock stats
export const mockStats: StatData = {
  totalLogs: 1000,
  totalAnomalies: mockAnomaliesEnhanced.length,
  criticalAnomalies: mockAnomaliesEnhanced.filter(a => a.severity === 'critical').length,
  highAnomalies: mockAnomaliesEnhanced.filter(a => a.severity === 'high').length,
  mediumAnomalies: mockAnomaliesEnhanced.filter(a => a.severity === 'medium').length,
  lowAnomalies: mockAnomaliesEnhanced.filter(a => a.severity === 'low').length,
  alertRate: parseFloat((mockAnomaliesEnhanced.length / 1000 * 100).toFixed(2)),
  avgConfidence: parseFloat((mockAnomaliesEnhanced.reduce((acc, curr) => acc + curr.confidence, 0) / mockAnomaliesEnhanced.length).toFixed(2)),
};

// Generate time series data for charts
export const generateTimeSeriesData = (days: number): TimeSeriesData[] => {
  return Array.from({ length: days }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (days - 1) + i);
    
    return {
      date: date.toISOString().split('T')[0],
      logs: Math.floor(Math.random() * 50) + 50,
      anomalies: Math.floor(Math.random() * 8) + 1,
    };
  });
};

// Generate real-time data stream simulation
export const generateRealtimeData = () => {
  const newLog: LogData = {
    id: `log-${Date.now()}`,
    timestamp: new Date().toISOString(),
    sourceIp: generateIp(),
    destinationIp: generateIp(),
    protocol: ['TCP', 'UDP', 'HTTP', 'HTTPS', 'FTP', 'SSH'][Math.floor(Math.random() * 6)],
    encrypted: Math.random() > 0.3,
    size: Math.floor(Math.random() * 5000) + 500,
  };
  
  const isAnomaly = Math.random() < 0.15;
  
  const newAnomaly = isAnomaly ? {
    id: `anomaly-${Date.now()}`,
    timestamp: new Date().toISOString(),
    severity: ['critical', 'high', 'medium', 'low'][Math.floor(Math.random() * 4)] as AnomalyData['severity'],
    sourceIp: newLog.sourceIp,
    destinationIp: newLog.destinationIp,
    protocol: newLog.protocol,
    action: ['blocked', 'allowed', 'flagged', 'quarantined'][Math.floor(Math.random() * 4)],
    confidence: Math.round(Math.random() * 100) / 100,
    reviewed: false,
    details: [
      'Unusual traffic pattern detected between hosts',
      'Multiple failed login attempts',
      'Suspicious outbound data transfer',
      'Port scanning activity',
      'Potential data exfiltration',
      'Unusual API access pattern',
      'Credential stuffing attempt',
      'Brute force attack detected',
    ][Math.floor(Math.random() * 8)],
  } : null;
  
  return { newLog, newAnomaly };
};
