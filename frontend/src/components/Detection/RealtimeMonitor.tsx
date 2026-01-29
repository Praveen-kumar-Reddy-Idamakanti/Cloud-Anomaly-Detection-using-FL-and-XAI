import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Switch } from '../ui/switch';
import { Label } from '../ui/label';
import { realtimeApi, RealtimeUpdate, DetectionResult } from '../../api/api';
import { EnhancedDetectionResults } from './EnhancedDetectionResults';

interface RealtimeMonitorProps {
  onDetectionUpdate?: (result: DetectionResult) => void;
}

export const RealtimeMonitor: React.FC<RealtimeMonitorProps> = ({
  onDetectionUpdate
}) => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isMonitoring, setIsMonitoring] = useState<boolean>(false);
  const [recentDetections, setRecentDetections] = useState<DetectionResult[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected');
  const [error, setError] = useState<string>('');
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (eventSourceRef.current) {
        realtimeApi.disconnectStream(eventSourceRef.current);
      }
    };
  }, []);

  const handleRealtimeUpdate = (update: RealtimeUpdate) => {
    console.log('Real-time update received:', update);
    
    switch (update.type) {
      case 'anomaly_detected':
        if (update.data && update.data.detectionResult) {
          const newResult: DetectionResult = {
            ...update.data.detectionResult,
            timestamp: new Date(update.data.detectionResult.timestamp)
          };
          
          setRecentDetections(prev => [newResult, ...prev.slice(0, 49)]); // Keep last 50
          onDetectionUpdate?.(newResult);
        }
        break;
        
      case 'model_update':
        setConnectionStatus('model updated');
        setTimeout(() => setConnectionStatus('connected'), 2000);
        break;
        
      case 'system_status':
        setConnectionStatus(update.data.status || 'unknown');
        break;
        
      default:
        console.log('Unknown update type:', update.type);
    }
  };

  const startMonitoring = () => {
    setError('');
    
    try {
      const eventSource = realtimeApi.connectToStream(handleRealtimeUpdate);
      eventSourceRef.current = eventSource;
      
      setIsConnected(true);
      setIsMonitoring(true);
      setConnectionStatus('connected');
      
      console.log('Real-time monitoring started');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start monitoring');
      setIsConnected(false);
      setIsMonitoring(false);
      setConnectionStatus('error');
    }
  };

  const stopMonitoring = () => {
    if (eventSourceRef.current) {
      realtimeApi.disconnectStream(eventSourceRef.current);
      eventSourceRef.current = null;
    }
    
    setIsConnected(false);
    setIsMonitoring(false);
    setConnectionStatus('disconnected');
    console.log('Real-time monitoring stopped');
  };

  const clearDetections = () => {
    setRecentDetections([]);
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'bg-green-500';
      case 'connecting': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      case 'model updated': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected';
      case 'connecting': return 'Connecting...';
      case 'error': return 'Connection Error';
      case 'model updated': return 'Model Updated';
      default: return 'Disconnected';
    }
  };

  const anomalyCount = recentDetections.filter(r => r.isAnomaly).length;

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Real-time Monitoring
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${getStatusColor()}`}></div>
              <span className="text-sm font-medium">{getStatusText()}</span>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Label htmlFor="monitoring-toggle">Enable Monitoring</Label>
                <Switch
                  id="monitoring-toggle"
                  checked={isMonitoring}
                  onCheckedChange={(checked) => {
                    if (checked) {
                      startMonitoring();
                    } else {
                      stopMonitoring();
                    }
                  }}
                />
              </div>
              
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={clearDetections}
                  disabled={recentDetections.length === 0}
                >
                  Clear History
                </Button>
                <Button
                  variant={isMonitoring ? "destructive" : "default"}
                  size="sm"
                  onClick={isMonitoring ? stopMonitoring : startMonitoring}
                >
                  {isMonitoring ? 'Stop' : 'Start'} Monitoring
                </Button>
              </div>
            </div>

            {/* Status Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{recentDetections.length}</div>
                <div className="text-sm text-gray-600">Total Detections</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">{anomalyCount}</div>
                <div className="text-sm text-gray-600">Anomalies</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{recentDetections.length - anomalyCount}</div>
                <div className="text-sm text-gray-600">Normal</div>
              </div>
            </div>

            {/* Connection Info */}
            <div className="text-sm text-gray-600">
              {isMonitoring ? (
                <span>🟢 Monitoring real-time network traffic for anomalies and attack types</span>
              ) : (
                <span>⚫ Real-time monitoring is disabled. Click "Start Monitoring" to begin.</span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recent Detections */}
      {recentDetections.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Recent Detections
              <Badge variant="outline">
                Last {Math.min(recentDetections.length, 10)} of {recentDetections.length}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <EnhancedDetectionResults 
              results={recentDetections.slice(0, 10)} 
              isLoading={false}
            />
          </CardContent>
        </Card>
      )}

      {/* Monitoring Instructions */}
      {!isMonitoring && (
        <Card>
          <CardHeader>
            <CardTitle>Real-time Monitoring Setup</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Alert>
                <AlertDescription>
                  <strong>Real-time monitoring requires:</strong>
                  <ul className="list-disc list-inside mt-2 space-y-1">
                    <li>Backend server running with SSE endpoint enabled</li>
                    <li>Continuous network data stream or periodic data generation</li>
                    <li>Stable internet connection for real-time updates</li>
                  </ul>
                </AlertDescription>
              </Alert>
              
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">How it works:</h4>
                <ol className="list-decimal list-inside space-y-1 text-sm">
                  <li>Server-Sent Events (SSE) establish a persistent connection</li>
                  <li>Backend sends real-time detection results as they occur</li>
                  <li>Frontend displays results instantly with attack type analysis</li>
                  <li>Connection automatically reconnects if interrupted</li>
                </ol>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
