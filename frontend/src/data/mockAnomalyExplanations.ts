// Mock data for 10 detailed anomaly explanations with different attack types
import { AnomalyData, ExplanationData } from '../api/api';

// Attack types configuration
const ATTACK_TYPES = [
  { id: 0, name: 'Botnet', description: 'Coordinated attack by botnet hosts', severity: 'high' as const, color: '#8b5cf6' },
  { id: 1, name: 'DoS', description: 'Denial of Service attack', severity: 'critical' as const, color: '#ef4444' },
  { id: 2, name: 'Infiltration', description: 'Unauthorized system access attempt', severity: 'high' as const, color: '#f97316' },
  { id: 3, name: 'Other', description: 'Other type of network attack', severity: 'medium' as const, color: '#eab308' },
  { id: 4, name: 'PortScan', description: 'Network port scanning activity', severity: 'medium' as const, color: '#f59e0b' }
];

// Generate random IP addresses
const generateIp = () => {
  return `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`;
};

// Generate random features (78 features for network traffic)
const generateFeatures = () => {
  return Array.from({ length: 78 }, () => Math.random());
};

// Generate feature importance data
const generateFeatureImportance = (attackType: number) => {
  const baseFeatures = [
    'Bwd PSH Flags', 'RST Flag Count', 'Fwd Seg Size Min', 'Fwd IAT Std', 
    'Fwd Packet Length Max', 'Subflow Bwd Packets', 'Bwd Bulk Rate Avg', 
    'Bwd Packet/Bulk Avg', 'Min Packet Length', 'Flow IAT Std'
  ];
  
  return baseFeatures.map((feature, index) => ({
    feature: feature,
    importance: Math.random() * 0.3 + 0.1,
    shap_value: Math.random() * 0.3 - 0.1,
    direction: Math.random() > 0.5 ? 'positive' : 'negative',
    impact: `${feature} contributed to ${ATTACK_TYPES[attackType].name.toLowerCase()} detection`
  }));
};

// Generate mock anomalies with detailed explanations
export const mockAnomalyExplanations: Array<{
  anomaly: AnomalyData;
  explanation: ExplanationData;
}> = Array.from({ length: 10 }, (_, index) => {
  const attackType = ATTACK_TYPES[index % 5];
  const sourceIp = generateIp();
  const destinationIp = generateIp();
  const protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS'];
  const actions = ['blocked', 'allowed', 'flagged'];
  const severities = ['low', 'medium', 'high', 'critical'] as const;
  
  const anomaly: AnomalyData = {
    id: `anomaly-${index + 1}`,
    timestamp: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
    sourceIp,
    destinationIp,
    protocol: protocols[Math.floor(Math.random() * protocols.length)] as any,
    action: actions[Math.floor(Math.random() * actions.length)] as any,
    severity: severities[Math.floor(Math.random() * severities.length)],
    confidence: Math.random() * 0.4 + 0.6, // 60-100%
    reviewed: Math.random() > 0.7,
    details: `Suspicious network traffic detected - potential ${attackType.name} attack`,
    anomalyScore: Math.random() * 0.5 + 0.2,
    reconstructionError: Math.random() * 0.5 + 0.2,
    attackType,
    attackConfidence: Math.random() * 0.4 + 0.6,
    features: JSON.stringify(generateFeatures())
  };

  const explanation: ExplanationData = {
    model_type: 'Autoencoder',
    explanation_type: 'comprehensive',
    feature_importances: generateFeatureImportance(attackType.id),
    note: `Comprehensive analysis for ${attackType.name} attack detection`,
    contributingFactors: [
      `Feature 8 contributed to anomaly detection`,
      `Unusual traffic patterns detected`,
      `Suspicious network behavior identified`
    ],
    recommendations: getAttackRecommendations(attackType.id),
    attack_type_explanation: {
      predicted_attack: attackType.name,
      confidence_reasoning: `Model confidence ${(anomaly.attackConfidence! * 100).toFixed(1)}% in prediction`,
      key_indicators: [
        'Flow Duration',
        'Bwd Bulk Rate Avg', 
        'Subflow Fwd Bytes'
      ]
    },
    attack_confidence: anomaly.attackConfidence,
    attack_feature_importances: generateFeatureImportance(attackType.id),
    reconstruction_error: anomaly.reconstructionError,
    phase3: {
      phase: 'phase3_classification',
      explanation_type: 'attack_type_explainability',
      features: generateFeatures(),
      attack_type: attackType.id,
      attack_name: attackType.name,
      confidence: anomaly.attackConfidence,
      explanation: {
        predicted_attack: attackType.name,
        confidence_reasoning: `Model confidence ${(anomaly.attackConfidence! * 100).toFixed(1)}% in prediction`,
        key_indicators: [
          'Flow Duration',
          'Bwd Bulk Rate Avg', 
          'Subflow Fwd Bytes'
        ]
      },
      timestamp: new Date().toISOString()
    }
  };

  return { anomaly, explanation };
});

// Individual anomaly data for easy access
export const getMockAnomalyById = (id: string) => {
  const mockData = mockAnomalyExplanations.find(item => item.anomaly.id === id);
  return mockData || null;
};

// Attack type specific recommendations
export const getAttackRecommendations = (attackType: number): string[] => {
  const recommendations = {
    0: [ // Botnet
      'Investigate potential Botnet command and control communication',
      'Monitor source IP for known malicious activity',
      'Check for coordinated traffic patterns from multiple sources',
      'Review network logs for unusual C2 protocols'
    ],
    1: [ // DoS
      'Immediate traffic rate limiting recommended',
      'Deploy DDoS mitigation measures',
      'Monitor for amplification attacks',
      'Check for SYN flood patterns'
    ],
    2: [ // Infiltration
      'Investigate potential unauthorized access attempts',
      'Review authentication logs for source IP',
      'Monitor for lateral movement attempts',
      'Check for data exfiltration patterns'
    ],
    3: [ // Other
      'Conduct detailed traffic analysis',
      'Review system logs for unusual patterns',
      'Monitor for emerging threat indicators',
      'Update threat intelligence feeds'
    ],
    4: [ // PortScan
      'Implement port knocking mechanisms',
      'Close unnecessary open ports',
      'Deploy intrusion detection systems',
      'Monitor for reconnaissance activity'
    ]
  };
  
  return recommendations[attackType as keyof typeof recommendations] || recommendations[3];
};

export default mockAnomalyExplanations;
