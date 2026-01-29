import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import { 
  Brain, 
  Target, 
  Zap, 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  TrendingUp,
  Activity
} from 'lucide-react';
import { DetectionResult, AttackTypeInfo } from '../../api/api';

interface TwoStagePredictionResultsProps {
  results: DetectionResult[];
  showDetails?: boolean;
  maxResults?: number;
}

export const TwoStagePredictionResults: React.FC<TwoStagePredictionResultsProps> = ({
  results,
  showDetails = true,
  maxResults = 20
}) => {
  const displayResults = results.slice(0, maxResults);
  
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getSeverityIcon = (attackType?: AttackTypeInfo) => {
    if (!attackType) return <CheckCircle className="h-4 w-4 text-green-500" />;
    
    switch (attackType.severity) {
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'high': return <XCircle className="h-4 w-4 text-orange-500" />;
      case 'medium': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'low': return <Shield className="h-4 w-4 text-blue-500" />;
      default: return <CheckCircle className="h-4 w-4 text-green-500" />;
    }
  };

  const formatTimestamp = (date: Date) => {
    return new Date(date).toLocaleString();
  };

  const getStageIcon = (stage: 'stage1' | 'stage2') => {
    return stage === 'stage1' ? 
      <Brain className="h-4 w-4 text-blue-500" /> : 
      <Target className="h-4 w-4 text-purple-500" />;
  };

  // Calculate statistics
  const totalSamples = displayResults.length;
  const anomalyCount = displayResults.filter(r => r.isAnomaly).length;
  const normalCount = totalSamples - anomalyCount;
  const attackTypeDistribution = displayResults.reduce((acc, result) => {
    if (result.isAnomaly && result.attackType) {
      const key = result.attackType.name;
      acc[key] = (acc[key] || 0) + 1;
    }
    return acc;
  }, {} as Record<string, number>);

  const averageConfidence = displayResults.reduce((sum, r) => sum + r.confidence, 0) / totalSamples;
  const averageAnomalyScore = displayResults.reduce((sum, r) => sum + r.anomalyScore, 0) / totalSamples;

  if (displayResults.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Brain className="h-5 w-5 mr-2" />
            Two-Stage Prediction Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No prediction results available</p>
            <p className="text-sm">Run a detection to see two-stage prediction results</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Statistics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center">
              <Brain className="h-5 w-5 mr-2" />
              Two-Stage Prediction Summary
            </div>
            <Badge variant="outline">
              {displayResults.length} samples
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{totalSamples}</div>
              <div className="text-sm text-gray-600">Total Samples</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{anomalyCount}</div>
              <div className="text-sm text-gray-600">Anomalies</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{normalCount}</div>
              <div className="text-sm text-gray-600">Normal</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {Object.keys(attackTypeDistribution).length}
              </div>
              <div className="text-sm text-gray-600">Attack Types</div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Average Confidence</span>
                <span className={getConfidenceColor(averageConfidence)}>
                  {(averageConfidence * 100).toFixed(1)}%
                </span>
              </div>
              <Progress value={averageConfidence * 100} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Average Anomaly Score</span>
                <span className="text-gray-600">
                  {averageAnomalyScore.toFixed(4)}
                </span>
              </div>
              <Progress value={(averageAnomalyScore / 0.5) * 100} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Attack Type Distribution */}
      {Object.keys(attackTypeDistribution).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Target className="h-5 w-5 mr-2" />
              Attack Type Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(attackTypeDistribution).map(([attackType, count]) => {
                const percentage = (count / anomalyCount) * 100;
                const attackTypeInfo = displayResults.find(r => r.attackType?.name === attackType)?.attackType;
                
                return (
                  <div key={attackType} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getSeverityIcon(attackTypeInfo)}
                      <div>
                        <div className="font-medium">{attackType}</div>
                        <div className="text-sm text-gray-600">
                          {attackTypeInfo?.description || 'Attack type detected'}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-medium">{count}</div>
                      <div className="text-sm text-gray-600">{percentage.toFixed(1)}%</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Detailed Results */}
      {showDetails && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center">
                <Zap className="h-5 w-5 mr-2" />
                Detailed Prediction Results
              </div>
              {results.length > maxResults && (
                <Button variant="outline" size="sm">
                  View All ({results.length})
                </Button>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {displayResults.map((result, index) => (
                <div 
                  key={result.id} 
                  className={`p-4 border rounded-lg ${
                    result.isAnomaly 
                      ? 'border-red-200 bg-red-50' 
                      : 'border-green-200 bg-green-50'
                  }`}
                >
                  {/* Header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      {getSeverityIcon(result.attackType)}
                      <Badge variant={result.isAnomaly ? "destructive" : "default"}>
                        {result.isAnomaly ? 'ANOMALY' : 'NORMAL'}
                      </Badge>
                      {result.attackType && (
                        <Badge 
                          variant="outline" 
                          style={{ 
                            borderColor: result.attackType.color, 
                            color: result.attackType.color 
                          }}
                        >
                          {result.attackType.name}
                        </Badge>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-500">
                        {formatTimestamp(result.timestamp)}
                      </div>
                    </div>
                  </div>

                  {/* Two-Stage Process */}
                  <div className="space-y-3">
                    {/* Stage 1: Anomaly Detection */}
                    <div className="flex items-start space-x-3">
                      {getStageIcon('stage1')}
                      <div className="flex-1">
                        <div className="font-medium text-sm">Stage 1: Anomaly Detection</div>
                        <div className="text-sm text-gray-600">
                          Autoencoder-based reconstruction error analysis
                        </div>
                        <div className="mt-2">
                          <div className="flex justify-between text-xs mb-1">
                            <span>Reconstruction Error</span>
                            <span className="font-mono">{result.anomalyScore.toFixed(4)}</span>
                          </div>
                          <Progress 
                            value={(result.anomalyScore / (result.threshold * 2)) * 100} 
                            className="h-2" 
                          />
                          <div className="text-xs text-gray-500 mt-1">
                            Threshold: {result.threshold.toFixed(4)}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Stage 2: Attack Classification (only for anomalies) */}
                    {result.isAnomaly && result.attackType && (
                      <div className="flex items-start space-x-3">
                        {getStageIcon('stage2')}
                        <div className="flex-1">
                          <div className="font-medium text-sm">Stage 2: Attack Type Classification</div>
                          <div className="text-sm text-gray-600">
                            Multi-class attack type identification
                          </div>
                          <div className="mt-2">
                            <div className="flex justify-between text-xs mb-1">
                              <span>Attack Confidence</span>
                              <span className="font-mono">
                                {((result.attackConfidence || 0) * 100).toFixed(1)}%
                              </span>
                            </div>
                            <Progress 
                              value={(result.attackConfidence || 0) * 100} 
                              className="h-2" 
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Additional Metrics */}
                  <div className="flex justify-between text-xs text-gray-500 pt-2 border-t">
                    <span>Features: {result.features.length} dimensions</span>
                    <span>Overall Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Insights */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <TrendingUp className="h-5 w-5 mr-2" />
            Performance Insights
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <Alert>
              <Brain className="h-4 w-4" />
              <AlertDescription>
                <strong>Two-Stage Model Performance:</strong> The model successfully identified 
                {anomalyCount} anomalies out of {totalSamples} samples ({((anomalyCount / totalSamples) * 100).toFixed(1)}%).
                {Object.keys(attackTypeDistribution).length > 0 && 
                  ` Attack classification achieved ${Object.keys(attackTypeDistribution).length} different attack types.`
                }
              </AlertDescription>
            </Alert>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <strong>Stage 1 (Anomaly Detection):</strong>
                <ul className="mt-1 space-y-1 text-gray-600">
                  <li>• Reconstruction error threshold: {displayResults[0]?.threshold.toFixed(4)}</li>
                  <li>• Average anomaly score: {averageAnomalyScore.toFixed(4)}</li>
                  <li>• Detection accuracy: {((normalCount / (normalCount + anomalyCount)) * 100).toFixed(1)}%</li>
                </ul>
              </div>
              <div>
                <strong>Stage 2 (Attack Classification):</strong>
                <ul className="mt-1 space-y-1 text-gray-600">
                  <li>• Attack types identified: {Object.keys(attackTypeDistribution).length}</li>
                  <li>• Classification confidence: {averageConfidence > 0.8 ? 'High' : averageConfidence > 0.6 ? 'Medium' : 'Low'}</li>
                  <li>• Most common attack: {Object.keys(attackTypeDistribution)[0] || 'N/A'}</li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
