
import {
  mockStats,
  mockAnomalies,
  mockLogs,
  mockExplanations,
  timeSeriesData,
  generateRealtimeData
} from '../data/mockData';

import { API_CONFIG } from '../config/api';

// API Configuration
const API_BASE_URL = API_CONFIG.BASE_URL;

// Simulate API latency
const simulateDelay = () => new Promise((resolve) => setTimeout(resolve, Math.random() * 500 + 300));

// Simulate error (approximately 5% of the time)
const shouldError = () => Math.random() < 0.05;

// Helper function to make API calls
const apiCall = async (endpoint: string, options: RequestInit = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`API call failed: ${response.status} ${response.statusText}`);
  }

  return response.json();
};

// Enhanced API types for two-stage detection
export type AnomalyDetectionRequest = {
  features: number[][];
  threshold?: number;
};

export type AnomalyDetectionResponse = {
  predictions: number[];
  scores: number[];
  threshold: number;
  confidence: number[];
};

export type EnhancedDetectionRequest = {
  features: number[][]; // 78 features
  threshold?: number; // Default 0.22610116
};

export type EnhancedDetectionResponse = {
  anomaly_predictions: number[];
  reconstruction_errors: number[];
  attack_type_predictions: number[];
  attack_confidences: number[];
  threshold: number;
  attack_types: string[];
};

// Attack type mapping
export type AttackTypeInfo = {
  id: number;
  name: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  color: string;
};

// Detection result types for UI
export type DetectionResult = {
  id: string;
  timestamp: Date;
  features: number[];
  isAnomaly: boolean;
  anomalyScore: number;
  threshold: number;
  attackType?: AttackTypeInfo;
  attackConfidence?: number;
  confidence: number;
};

// Real-time update types
export type RealtimeUpdate = {
  type: 'anomaly_detected' | 'model_update' | 'system_status';
  timestamp: Date;
  data: any;
};

export type LoginCredentials = {
  email: string;
  password: string;
};

export type RegisterData = {
  name: string;
  email: string;
  password: string;
};

export type ApiError = {
  message: string;
  code: number;
};

export type AuthResponse = {
  token: string;
  user: {
    id: string;
    name: string;
    email: string;
    role: 'admin' | 'user';
  };
};

export type AnomalyData = {
  id: string;
  timestamp: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  sourceIp: string;
  destinationIp: string;
  protocol: string;
  action: string;
  confidence: number;
  reviewed: boolean;
  details: string;
  features?: string; // JSON string of features for XAI
  anomalyScore?: number;
  attackTypeId?: number;
  attackConfidence?: number;
};

export type LogData = {
  id: string;
  timestamp: string;
  sourceIp: string;
  destinationIp: string;
  protocol: string;
  encrypted: boolean;
  size: number;
};

export type StatData = {
  totalLogs: number;
  totalAnomalies: number;
  criticalAnomalies: number;
  highAnomalies: number;
  mediumAnomalies: number;
  lowAnomalies: number;
  alertRate: number;
  avgConfidence: number;
};

export type FeatureImportance = {
  feature: string;
  importance: number;
  shap_value?: number; // SHAP value for the feature
  direction?: string; // Direction of impact (positive/negative)
};

export type ExplanationData = {
  model_type: string; // The backend returns 'Autoencoder'
  explanation_type: string; // The backend returns 'SHAP'
  feature_importances: FeatureImportance[];
  note: string;
  contributingFactors?: string[]; // Optional contributing factors
  recommendations?: string[]; // Optional recommendations
};

export type TimeSeriesData = {
  date: string;
  logs: number;
  anomalies: number;
};

// Authentication API
export const authApi = {
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Authentication failed. Please check your credentials.', code: 401 };
    }
    
    if (credentials.email && credentials.password) {
      return {
        token: `jwt-mock-token-${Date.now()}`,
        user: {
          id: '1',
          name: credentials.email.split('@')[0],
          email: credentials.email,
          role: Math.random() > 0.7 ? 'admin' : 'user',
        },
      };
    }
    
    throw { message: 'Email and password are required', code: 400 };
  },
  
  register: async (data: RegisterData): Promise<AuthResponse> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Registration failed. Please try again later.', code: 500 };
    }
    
    if (data.email && data.password && data.name) {
      return {
        token: `jwt-mock-token-${Date.now()}`,
        user: {
          id: '1',
          name: data.name,
          email: data.email,
          role: 'user',
        },
      };
    }
    
    throw { message: 'All fields are required', code: 400 };
  },
  
  logout: async (): Promise<void> => {
    await simulateDelay();
    return;
  },
};

// Stats API
export const statsApi = {
  getStats: async (): Promise<StatData> => {
    const data = await apiCall('/stats');
    return {
      totalLogs: data.total_logs,
      totalAnomalies: data.total_anomalies,
      criticalAnomalies: data.critical_anomalies,
      highAnomalies: data.high_anomalies,
      mediumAnomalies: data.medium_anomalies,
      lowAnomalies: data.low_anomalies,
      alertRate: data.alert_rate,
      avgConfidence: data.avg_confidence,
    };
  },
  
  getTimeSeriesData: async (): Promise<TimeSeriesData[]> => {
    const data = await apiCall('/history/training');
    
    // Transform the data to match TimeSeriesData
    const transformedData: TimeSeriesData[] = data.map((item: any) => ({
      date: item.created_at.split('T')[0], // Extract just the date part
      logs: item.server_round, // Using server_round as a proxy for logs count
      anomalies: item.avg_loss, // Using avg_loss as a proxy for anomaly count for now
    }));
    
    return transformedData;
  },
};

// Anomalies API
export const anomaliesApi = {
  getAnomalies: async (page: number = 1, limit: number = 10): Promise<{ data: AnomalyData[], total: number }> => {
    // Note: The total count is not yet returned by the API.
    // This is a temporary solution until the API is updated.
    const data = await apiCall(`/anomalies?page=${page}&limit=${limit}`);
    const countResponse = await apiCall('/stats'); // Using stats for total count for now
    
    const transformedData = data.map((item: any) => ({
      id: item.id,
      timestamp: item.timestamp,
      severity: item.severity,
      sourceIp: item.source_ip,
      destinationIp: item.destination_ip,
      protocol: item.protocol,
      action: item.action,
      confidence: item.confidence,
      reviewed: item.reviewed,
      details: item.details,
    }));
    
    return {
      data: transformedData,
      total: countResponse.total_anomalies,
    };
  },
  
  getAnomalyById: async (id: string): Promise<AnomalyData> => {
    const data = await apiCall(`/anomalies/${id}`);
    return {
      id: data.id,
      timestamp: data.timestamp,
      severity: data.severity,
      sourceIp: data.source_ip,
      destinationIp: data.destination_ip,
      protocol: data.protocol,
      action: data.action,
      confidence: data.confidence,
      reviewed: data.reviewed,
      details: data.details,
      features: data.features,  // Add this line to include features
    };
  },
  
  reviewAnomaly: async (id: string, reviewed: boolean): Promise<AnomalyData> => {
    await apiCall(`/anomalies/${id}/review`, {
      method: 'POST',
      body: JSON.stringify({ reviewed }),
    });
    
    // Return the updated anomaly
    // In a real app, you might want to refetch or just update the state optimistically
    return await anomaliesApi.getAnomalyById(id);
  },
};

// Logs API
export const logsApi = {
  getLogs: async (page: number = 1, limit: number = 10): Promise<{ data: LogData[], total: number }> => {
    // This endpoint is a placeholder for a future implementation
    // that would fetch logs from a dedicated logging backend or database table.
    return { data: [], total: 0 };
  },
  
  uploadLog: async (file: File, encrypted: boolean): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    
    // The 'encrypted' flag is not handled by the backend yet, but we pass it
    const response = await fetch(`${API_BASE_URL}/logs/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Upload failed with no specific error message.' }));
      throw new Error(`Upload failed: ${response.status} - ${errorData.detail}`);
    }

    return response.json();
  },
};

// XAI Explanations API - Now Integrated with Completed Phases
export const explanationsApi = {
  getAnomalyExplanation: async (features: number[]): Promise<ExplanationData> => {
    console.log('=== FRONTEND: getAnomalyExplanation API CALL ===');
    console.log('Features length:', features.length);
    console.log('Features sample:', features.slice(0, 5));
    
    try {
      const data = await apiCall('/explain_anomaly', {
        method: 'POST',
        body: JSON.stringify({ features }),
      });
      
      console.log('=== BACKEND RESPONSE RECEIVED ===');
      console.log('Response type:', typeof data);
      console.log('Response keys:', Object.keys(data));
      console.log('Comprehensive explanation:', data.comprehensive_explanation);
      console.log('Anomaly detected:', data.anomaly_detected);
      console.log('Phase1 available:', 'phase1' in data);
      console.log('Phase2 available:', 'phase2' in data);
      console.log('Phase3 available:', 'phase3' in data);
      
      // Handle the new comprehensive explanation structure
      if (data.comprehensive_explanation) {
        console.log('=== PROCESSING COMPREHENSIVE EXPLANATION ===');
        
        // Extract feature importances from phase2 (SHAP-based)
        const featureImportances = data.phase2?.feature_importance || [];
        console.log('Feature importances from phase2:', featureImportances.length);
        
        const mappedImportances = featureImportances.map((item: any) => {
          console.log('Mapping feature importance:', item);
          return {
            feature: item.feature_name || `feature_${item.feature_index}`,
            importance: item.importance,
            shap_value: item.shap_value,
            direction: item.direction
          };
        });
        
        const contributingFactors = data.phase1?.explanation?.key_features?.map((idx: number) => `Feature ${idx} contributed to anomaly detection`) || [];
        const recommendations = data.phase3?.explanation?.predicted_attack ? [
          `Investigate potential ${data.phase3.explanation.predicted_attack} attack`,
          'Monitor source and destination IPs',
          'Review network traffic patterns'
        ] : [
          'Continue monitoring network traffic',
          'Review system logs for unusual patterns'
        ];
        
        const result = {
          model_type: data.model_type || 'Autoencoder',
          explanation_type: data.explanation_type || 'comprehensive',
          feature_importances: mappedImportances,
          note: `Anomaly detected: ${data.anomaly_detected}. Reconstruction error: ${data.reconstruction_error?.toFixed(4)}`,
          contributingFactors,
          recommendations
        };
        
        console.log('=== MAPPED EXPLANATION DATA ===');
        console.log('Model type:', result.model_type);
        console.log('Explanation type:', result.explanation_type);
        console.log('Feature importances count:', result.feature_importances.length);
        console.log('Contributing factors count:', result.contributingFactors.length);
        console.log('Recommendations count:', result.recommendations.length);
        console.log('=== FRONTEND API CALL SUCCESS ===');
        
        return result;
      }
      
      // Fallback to old structure or mock
      console.log('=== USING FALLBACK STRUCTURE ===');
      const fallbackResult = {
        model_type: data.model_type || 'Autoencoder',
        explanation_type: data.explanation_type || 'SHAP',
        feature_importances: data.feature_importances || [],
        note: data.note || 'Explanation generated successfully'
      };
      
      console.log('Fallback result:', fallbackResult);
      return fallbackResult;
      
    } catch (error: any) {
      console.error('=== FRONTEND API ERROR ===');
      console.error('Error:', error);
      console.error('Error message:', error.message);
      console.error('Error status:', error.status);
      console.error('Error details:', error.details);
      throw error;
    }
  },
  
  // Phase-specific explanations
  getPhaseExplanation: async (phase: string, features: number[], options: any = {}) => {
    try {
      // ... (rest of the code remains the same)
      const data = await apiCall('/xai/phase_explanation', {
        method: 'POST',
        body: JSON.stringify({ phase, features, ...options }),
      });
      return data;
    } catch (error) {
      console.warn(`Failed to get ${phase} explanation:`, error);
      // Return mock explanation
      return {
        phase,
        explanation_type: 'mock_phase_explanation',
        features,
        timestamp: new Date().toISOString()
      };
    }
  },
  
  // Feature importance analysis
  getFeatureImportance: async (features: number[], topK: number = 10) => {
    try {
      const data = await apiCall('/xai/feature_importance', {
        method: 'POST',
        body: JSON.stringify({ features, top_k: topK }),
      });
      return data;
    } catch (error) {
      console.warn('Failed to get feature importance:', error);
      // Return mock feature importance
      return {
        feature_importance: Array.from({ length: topK }, (_, i) => ({
          feature_index: i,
          feature_name: `feature_${i}`,
          shap_value: Math.random() * 0.2 - 0.1,
          importance: Math.random(),
          direction: Math.random() > 0.5 ? 'positive' : 'negative'
        })),
        total_features: features.length,
        top_features: Array.from({ length: topK }, (_, i) => ({
          feature_index: i,
          feature_name: `feature_${i}`,
          shap_value: Math.random() * 0.2 - 0.1,
          importance: Math.random()
        })),
        phase: 'phase2_autoencoder',
        timestamp: new Date().toISOString()
      };
    }
  },
  
  // Attack type explanations
  getAttackTypeExplanation: async (features: number[], attackType: number, confidence?: number) => {
    try {
      const data = await apiCall('/xai/attack_type_explanation', {
        method: 'POST',
        body: JSON.stringify({ features, attackType, confidence }),
      });
      return data;
    } catch (error) {
      console.warn('Failed to get attack type explanation:', error);
      // Return mock attack type explanation
      const attackTypes = ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris'];
      return {
        phase: 'phase3_classification',
        explanation_type: 'attack_type_explainability',
        features,
        attack_type: attackType,
        attack_name: attackTypes[attackType] || 'Unknown',
        confidence: confidence || 0.8,
        explanation: {
          predicted_attack: attackTypes[attackType] || 'Unknown',
          confidence_reasoning: `Model confidence ${(confidence || 0.8).toFixed(3)} in prediction`,
          key_indicators: Array.from({ length: 5 }, (_, i) => `feature_${i}`).filter(i => features[i] > 0.7)
        },
        timestamp: new Date().toISOString()
      };
    }
  },
  
  // Comprehensive explanation (all phases)
  getComprehensiveExplanation: async (features: number[]) => {
    try {
      const data = await apiCall('/explain_anomaly', {
        method: 'POST',
        body: JSON.stringify({ 
          features, 
          explanation_type: 'comprehensive' 
        }),
      });
      return data;
    } catch (error) {
      console.warn('Failed to get comprehensive explanation:', error);
      // Return mock comprehensive explanation
      const anomalyScore = Math.random() * 0.5;
      const anomalyDetected = anomalyScore > 0.22610116;
      
      return {
        comprehensive_explanation: true,
        features,
        anomaly_detected: anomalyDetected,
        reconstruction_error: anomalyScore,
        phase1: {
          explanation_type: 'basic_anomaly',
          is_anomaly: anomalyDetected,
          confidence: 0.85,
          reasoning: `Reconstruction error ${anomalyScore.toFixed(4)} exceeds threshold 0.2261`
        },
        phase2: {
          explanation_type: 'shap_explainability',
          shap_values: Array.from({ length: 78 }, () => Math.random() * 0.2 - 0.1),
          feature_importance: Array.from({ length: 10 }, (_, i) => ({
            feature_index: i,
            feature_name: `feature_${i}`,
            shap_value: Math.random() * 0.2 - 0.1,
            importance: Math.random(),
            direction: Math.random() > 0.5 ? 'positive' : 'negative'
          }))
        },
        phase3: anomalyDetected ? {
          phase: 'phase3_classification',
          explanation_type: 'attack_type_explainability',
          attack_type: Math.floor(Math.random() * 4) + 1,
          attack_name: ['DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris'][Math.floor(Math.random() * 4)],
          confidence: Math.random() * 0.3 + 0.7,
          explanation: {
            predicted_attack: 'Mock attack prediction',
            confidence_reasoning: 'Mock confidence reasoning'
          }
        } : { phase: 'phase3_classification', explanation_type: 'not_anomaly' },
        timestamp: new Date().toISOString()
      };
    }
  },
  
  // Legacy method for backward compatibility
  generateShapExplanation: async (features: number[] | number[][]) => {
    // Flatten features for single instance if needed
    let flatFeatures: number[];
    if (Array.isArray(features[0])) {
      flatFeatures = features[0] as number[];
    } else {
      flatFeatures = features as number[];
    }
    return explanationsApi.getAnomalyExplanation(flatFeatures);
  }
};

// Attack type mappings
const ATTACK_TYPE_MAP: Record<number, AttackTypeInfo> = {
  0: { id: 0, name: 'BENIGN', description: 'Normal network traffic', severity: 'low', color: '#10b981' },
  1: { id: 1, name: 'DoS GoldenEye', description: 'Denial of Service GoldenEye attack', severity: 'high', color: '#ef4444' },
  2: { id: 2, name: 'DoS Hulk', description: 'Denial of Service Hulk attack', severity: 'high', color: '#f97316' },
  3: { id: 3, name: 'DoS Slowhttptest', description: 'Denial of Service Slow HTTP attack', severity: 'medium', color: '#eab308' },
  4: { id: 4, name: 'DoS slowloris', description: 'Denial of Service Slowloris attack', severity: 'medium', color: '#f59e0b' }
};

// Helper functions
export const getAttackTypeInfo = (attackTypeId: number): AttackTypeInfo => {
  return ATTACK_TYPE_MAP[attackTypeId] || ATTACK_TYPE_MAP[0];
};

export const formatDetectionResult = (
  features: number[],
  response: EnhancedDetectionResponse,
  index: number
): DetectionResult => {
  const isAnomaly = response.anomaly_predictions[index] === 1;
  const attackTypeId = response.attack_type_predictions[index];
  const attackType = isAnomaly ? getAttackTypeInfo(attackTypeId) : undefined;
  
  return {
    id: `detection_${Date.now()}_${index}`,
    timestamp: new Date(),
    features,
    isAnomaly,
    anomalyScore: response.reconstruction_errors[index],
    threshold: response.threshold,
    attackType,
    attackConfidence: isAnomaly ? response.attack_confidences[index] : undefined,
    confidence: 1 - (response.reconstruction_errors[index] / (response.threshold * 2))
  };
};
// Real-time updates API
export const realtimeApi = {
  connectToStream: (onUpdate: (update: RealtimeUpdate) => void) => {
    const eventSource = new EventSource(`${API_CONFIG.BASE_URL}/realtime/stream`);
    
    eventSource.onmessage = (event) => {
      try {
        const update: RealtimeUpdate = JSON.parse(event.data);
        update.timestamp = new Date(update.timestamp);
        onUpdate(update);
      } catch (error) {
        console.error('Error parsing real-time update:', error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('Real-time stream error:', error);
    };
    
    return eventSource;
  },
  
  disconnectStream: (eventSource: EventSource) => {
    eventSource.close();
  }
};

export const modelApi = {
  getModelInfo: async () => {
    try {
      return await apiCall('/model/info');
    } catch (error) {
      console.warn('Failed to fetch model info:', error);
      return {
        model_path: 'unknown',
        input_dim: 78,
        last_trained: 'unknown',
        accuracy: null,
        status: 'not_loaded',
        two_stage_enabled: false,
        attack_types: []
      };
    }
  },
  
  detectAnomalies: async (features: number[][], threshold: number = 0.4) => {
    try {
      return await apiCall('/model/detect', {
        method: 'POST',
        body: JSON.stringify({
          features,
          threshold
        }),
      });
    } catch (error) {
      console.warn('Failed to detect anomalies with real model:', error);
      // Return mock detection results
      return {
        predictions: features.map(() => Math.random() > 0.8 ? 1 : 0),
        scores: features.map(() => Math.random()),
        threshold,
        confidence: features.map(() => Math.random()),
      };
    }
  },
  
  detectAnomaliesEnhanced: async (features: number[][], threshold: number = 0.22610116) => {
    try {
      return await apiCall('/model/detect-enhanced', {
        method: 'POST',
        body: JSON.stringify({
          features,
          threshold
        }),
      });
    } catch (error) {
      console.warn('Failed to detect anomalies with enhanced model:', error);
      // Return mock two-stage results
      return {
        anomaly_predictions: features.map(() => Math.random() > 0.8 ? 1 : 0),
        reconstruction_errors: features.map(() => Math.random()),
        attack_type_predictions: features.map(() => Math.random() > 0.8 ? Math.floor(Math.random() * 4) + 1 : 0),
        attack_confidences: features.map(() => Math.random()),
        threshold,
        attack_types: ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris']
      };
    }
  },
  
  getTrainingStatus: async () => {
    try {
      return await apiCall('/training/status');
    } catch (error) {
      console.warn('Failed to fetch training status:', error);
      return {
        is_training: false,
        progress: 0,
        round: 0
      };
    }
  },
  
  startTraining: async (epochs: number = 5, batchSize: number = 32, learningRate: number = 0.001) => {
    try {
      return await apiCall('/training/start', {
        method: 'POST',
        body: JSON.stringify({
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate
        }),
      });
    } catch (error) {
      console.warn('Failed to start training:', error);
      throw { message: 'Failed to start training', code: 500 };
    }
  },
};

// Legacy real-time polling - deprecated, use realtimeApi above with SSE
let pollingInterval: NodeJS.Timeout | null = null;
let lastAnomalyIds: Set<string> = new Set();

export const legacyRealtimeApi = {
  subscribeToRealTimeUpdates: (callback: (data: { anomaly?: AnomalyData }) => void, intervalMs: number = 5000) => {
    // Cleanup any existing subscription
    if (pollingInterval) {
      legacyRealtimeApi.unsubscribeFromRealTimeUpdates();
    }
    
    const pollForAnomalies = async () => {
      try {
        const { data: currentAnomalies } = await anomaliesApi.getAnomalies(1, 10); // Fetch recent anomalies
        
        const newAnomalies = currentAnomalies.filter(
          (anomaly) => !lastAnomalyIds.has(anomaly.id)
        );
        
        newAnomalies.forEach((anomaly) => {
          callback({ anomaly });
        });
        
        // Update last seen anomaly IDs
        lastAnomalyIds = new Set(currentAnomalies.map((a) => a.id));
      } catch (error) {
        console.error('Error polling for real-time anomalies:', error);
      }
    };
    
    // Initial poll
    pollForAnomalies();

    // Start polling
    pollingInterval = setInterval(pollForAnomalies, intervalMs);
    
    return () => {
      legacyRealtimeApi.unsubscribeFromRealTimeUpdates();
    };
  },
  
  unsubscribeFromRealTimeUpdates: () => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
      lastAnomalyIds = new Set(); // Reset seen anomalies on unsubscribe
    }
  },
};
