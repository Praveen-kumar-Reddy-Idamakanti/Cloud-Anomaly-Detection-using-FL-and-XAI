import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Switch } from '../ui/switch';
import { AlertTriangle, Activity, Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { realtimeApi, RealtimeUpdate, DetectionResult, getAttackTypeInfo } from '../../api/api';

interface RealTimeAnomalyFeedProps {
  maxItems?: number;
  showControls?: boolean;
}

export const RealTimeAnomalyFeed: React.FC<RealTimeAnomalyFeedProps> = ({
  maxItems = 10,
  showControls = true
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [recentDetections, setRecentDetections] = useState<DetectionResult[]>([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        realtimeApi.disconnectStream(eventSourceRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (autoRefresh && !isMonitoring) {
      // Start monitoring automatically if auto-refresh is enabled
      startMonitoring();
    } else if (!autoRefresh && isMonitoring) {
      // Stop monitoring if auto-refresh is disabled
      stopMonitoring();
    }
  }, [autoRefresh]);

  const handleRealtimeUpdate = (update: RealtimeUpdate) => {
    switch (update.type) {
      case 'anomaly_detected':
        if (update.data && update.data.detectionResult) {
          const newResult: DetectionResult = {
            ...update.data.detectionResult,
            timestamp: new Date(update.data.detectionResult.timestamp)
          };
          
          setRecentDetections(prev => {
            const updated = [newResult, ...prev];
            return updated.slice(0, maxItems);
          });
        }
        break;
        
      case 'model_update':
        setConnectionStatus('model updated');
        setTimeout(() => setConnectionStatus('connected'), 2000);
        break;
        
      case 'system_status':
        setConnectionStatus(update.data.status || 'unknown');
        break;
    }
  };

  const startMonitoring = () => {
    try {
      const eventSource = realtimeApi.connectToStream(handleRealtimeUpdate);
      eventSourceRef.current = eventSource;
      
      setIsConnected(true);
      setIsMonitoring(true);
      setConnectionStatus('connected');
    } catch (error) {
      console.error('Failed to start monitoring:', error);
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
  };

  const handleToggleMonitoring = () => {
    if (isMonitoring) {
      stopMonitoring();
    } else {
      startMonitoring();
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-600';
      case 'connecting': return 'text-yellow-600';
      case 'error': return 'text-red-600';
      case 'model updated': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return <Wifi className="h-4 w-4" />;
      case 'connecting': return <RefreshCw className="h-4 w-4 animate-spin" />;
      case 'error': return <WifiOff className="h-4 w-4" />;
      default: return <WifiOff className="h-4 w-4" />;
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

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return `${seconds}s ago`;
  };

  // Generate some mock data for demonstration
  useEffect(() => {
    if (!isMonitoring && recentDetections.length === 0) {
      // Add some mock data for initial display
      const mockData: DetectionResult[] = [
        {
          id: 'mock_1',
          timestamp: new Date(Date.now() - 60000),
          features: Array.from({ length: 78 }, () => Math.random()),
          isAnomaly: true,
          anomalyScore: 0.85,
          threshold: 0.22610116,
          attackType: getAttackTypeInfo(2),
          attackConfidence: 0.92,
          confidence: 0.85
        },
        {
          id: 'mock_2',
          timestamp: new Date(Date.now() - 120000),
          features: Array.from({ length: 78 }, () => Math.random()),
          isAnomaly: false,
          anomalyScore: 0.15,
          threshold: 0.22610116,
          confidence: 0.95
        },
        {
          id: 'mock_3',
          timestamp: new Date(Date.now() - 180000),
          features: Array.from({ length: 78 }, () => Math.random()),
          isAnomaly: true,
          anomalyScore: 0.73,
          threshold: 0.22610116,
          attackType: getAttackTypeInfo(1),
          attackConfidence: 0.87,
          confidence: 0.73
        }
      ];
      setRecentDetections(mockData);
    }
  }, [isMonitoring, recentDetections.length]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center text-lg">
            <Activity className="h-5 w-5 mr-2" />
            Real-time Anomaly Feed
          </CardTitle>
          {showControls && (
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-2 text-sm">
                <Switch
                  checked={autoRefresh}
                  onCheckedChange={setAutoRefresh}
                  disabled={connectionStatus === 'connecting'}
                />
                <span className="text-xs text-gray-600">Auto</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleToggleMonitoring}
                disabled={connectionStatus === 'connecting'}
              >
                {isMonitoring ? 'Pause' : 'Start'}
              </Button>
            </div>
          )}
        </div>
        <div className={`flex items-center space-x-2 text-sm ${getStatusColor()}`}>
          {getStatusIcon()}
          <span>{getStatusText()}</span>
          {recentDetections.length > 0 && (
            <span className="text-gray-500">
              ({recentDetections.length} recent)
            </span>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="pt-0">
        {recentDetections.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">
              {isMonitoring ? 'Waiting for detections...' : 'No recent detections'}
            </p>
            {!isMonitoring && (
              <Button variant="outline" size="sm" className="mt-2" onClick={startMonitoring}>
                Start Monitoring
              </Button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {recentDetections.map((detection) => (
              <div
                key={detection.id}
                className={`p-3 border rounded-lg transition-colors ${
                  detection.isAnomaly 
                    ? 'border-red-200 bg-red-50 hover:bg-red-100' 
                    : 'border-green-200 bg-green-50 hover:bg-green-100'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <AlertTriangle className={`h-4 w-4 ${
                        detection.isAnomaly ? 'text-red-500' : 'text-green-500'
                      }`} />
                      <Badge 
                        variant={detection.isAnomaly ? "destructive" : "default"}
                        className="text-xs"
                      >
                        {detection.isAnomaly ? 'ANOMALY' : 'NORMAL'}
                      </Badge>
                      {detection.attackType && (
                        <Badge 
                          variant="outline" 
                          className="text-xs"
                          style={{ 
                            borderColor: detection.attackType.color, 
                            color: detection.attackType.color 
                          }}
                        >
                          {detection.attackType.name}
                        </Badge>
                      )}
                    </div>
                    
                    <div className="text-xs text-gray-600 space-y-1">
                      <div className="flex justify-between">
                        <span>Score: {detection.anomalyScore.toFixed(4)}</span>
                        <span>Confidence: {(detection.confidence * 100).toFixed(1)}%</span>
                      </div>
                      {detection.attackConfidence && (
                        <div className="flex justify-between">
                          <span>Attack Confidence: {(detection.attackConfidence * 100).toFixed(1)}%</span>
                          <span>{formatTimestamp(detection.timestamp)}</span>
                        </div>
                      )}
                      {!detection.attackConfidence && (
                        <div className="text-right">
                          <span>{formatTimestamp(detection.timestamp)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {detection.isAnomaly && (
                    <div className="ml-2">
                      <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {recentDetections.length >= maxItems && (
              <div className="text-center pt-2">
                <Button variant="ghost" size="sm" className="text-xs">
                  View all detections →
                </Button>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
