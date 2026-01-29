import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Alert, AlertDescription } from '../ui/alert';
import { DetectionResult, AttackTypeInfo } from '../../api/api';

interface EnhancedDetectionResultsProps {
  results: DetectionResult[];
  isLoading?: boolean;
}

export const EnhancedDetectionResults: React.FC<EnhancedDetectionResultsProps> = ({
  results,
  isLoading = false
}) => {
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
    return new Date(date).toLocaleString();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Detection Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Analyzing network traffic...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (results.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Detection Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            No detection results available. Run an analysis to see results.
          </div>
        </CardContent>
      </Card>
    );
  }

  const anomalyCount = results.filter(r => r.isAnomaly).length;
  const normalCount = results.length - anomalyCount;

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Detection Summary
            <Badge variant={anomalyCount > 0 ? "destructive" : "default"}>
              {anomalyCount} Anomalies Detected
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{results.length}</div>
              <div className="text-sm text-gray-600">Total Samples</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{normalCount}</div>
              <div className="text-sm text-gray-600">Normal Traffic</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{anomalyCount}</div>
              <div className="text-sm text-gray-600">Anomalies</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Results */}
      <div className="space-y-4">
        {results.map((result, index) => (
          <Card key={result.id} className={result.isAnomaly ? 'border-red-200' : 'border-green-200'}>
            <CardContent className="pt-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Badge variant={result.isAnomaly ? "destructive" : "default"}>
                    {result.isAnomaly ? 'ANOMALY' : 'NORMAL'}
                  </Badge>
                  <span className="text-sm text-gray-500">
                    {formatTimestamp(result.timestamp)}
                  </span>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-medium ${getConfidenceColor(result.confidence)}`}>
                    {(result.confidence * 100).toFixed(1)}% Confidence
                  </div>
                </div>
              </div>

              {/* Anomaly Score */}
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-1">
                  <span>Anomaly Score</span>
                  <span>{result.anomalyScore.toFixed(4)}</span>
                </div>
                <Progress 
                  value={(result.anomalyScore / (result.threshold * 2)) * 100} 
                  className="h-2"
                />
                <div className="text-xs text-gray-500 mt-1">
                  Threshold: {result.threshold.toFixed(4)}
                </div>
              </div>

              {/* Attack Type Information */}
              {result.isAnomaly && result.attackType && (
                <Alert className="mb-4">
                  <AlertDescription>
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">{result.attackType.name}</div>
                        <div className="text-sm text-gray-600">{result.attackType.description}</div>
                      </div>
                      <div className="text-right">
                        <Badge className={getSeverityColor(result.attackType.severity)}>
                          {result.attackType.severity.toUpperCase()}
                        </Badge>
                        {result.attackConfidence && (
                          <div className="text-sm mt-1">
                            Attack Confidence: {(result.attackConfidence * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>
                    </div>
                  </AlertDescription>
                </Alert>
              )}

              {/* Feature Summary */}
              <div className="text-xs text-gray-500">
                Features: {result.features.length} dimensions analyzed
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};
