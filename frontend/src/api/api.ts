
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
};

export type ExplanationData = {
  model_type: string; // The backend returns 'Autoencoder'
  explanation_type: string; // The backend returns 'SHAP'
  feature_importances: FeatureImportance[];
  note: string;
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

// XAI Explanations API
export const explanationsApi = {
  getAnomalyExplanation: async (features: number[]): Promise<ExplanationData> => {
    const data = await apiCall('/explain_anomaly', {
      method: 'POST',
      body: JSON.stringify({ features }),
    });
    
    return {
      model_type: data.model_type,
      explanation_type: data.explanation_type,
      feature_importances: data.feature_importances,
      note: data.note,
    };
  },
};

// Model Management API
export const modelApi = {
  getModelInfo: async () => {
    try {
      return await apiCall('/model/info');
    } catch (error) {
      console.warn('Failed to fetch model info:', error);
      return {
        model_path: 'unknown',
        input_dim: 9,
        last_trained: 'unknown',
        accuracy: null,
        status: 'not_loaded'
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

// Real-time data simulation - currently using polling as `/realtime/stream` is a placeholder.
// For true real-time, consider Server-Sent Events (SSE) or WebSockets.
let pollingInterval: NodeJS.Timeout | null = null;
let lastAnomalyIds: Set<string> = new Set();

export const realtimeApi = {
  subscribeToRealTimeUpdates: (callback: (data: { anomaly?: AnomalyData }) => void, intervalMs: number = 5000) => {
    // Cleanup any existing subscription
    if (pollingInterval) {
      realtimeApi.unsubscribeFromRealTimeUpdates();
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
      realtimeApi.unsubscribeFromRealTimeUpdates();
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
