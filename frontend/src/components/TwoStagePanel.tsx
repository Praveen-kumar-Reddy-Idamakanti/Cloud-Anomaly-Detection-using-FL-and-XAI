import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Shield, AlertTriangle, CheckCircle } from 'lucide-react';
import { TwoStageDetectionResult } from '../api/api';

interface TwoStagePanelProps {
  anomalyResult: TwoStageDetectionResult;
}

const TwoStagePanel: React.FC<TwoStagePanelProps> = ({ anomalyResult }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Shield className="mr-2 h-5 w-5 text-blue-500" />
          Two-Stage Detection Pipeline
        </CardTitle>
        <CardDescription>
          Anomaly detection followed by attack type classification
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Stage 1: Anomaly Detection */}
          <div className="border-l-4 border-green-500 pl-4">
            <div className="flex items-center mb-2">
              <h4 className="font-medium text-sm">Stage 1: Anomaly Detection</h4>
              {anomalyResult.isAnomaly ? (
                <AlertTriangle className="h-4 w-4 text-red-500 ml-2" />
              ) : (
                <CheckCircle className="h-4 w-4 text-green-500 ml-2" />
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Result:</span>
                <span className={`ml-2 font-medium ${anomalyResult.isAnomaly ? 'text-red-500' : 'text-green-500'}`}>
                  {anomalyResult.isAnomaly ? 'Anomaly Detected' : 'Normal Traffic'}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Score:</span>
                <span className="ml-2 font-mono">{anomalyResult.anomalyScore?.toFixed(4) || 'N/A'}</span>
              </div>
              {anomalyResult.reconstructionError !== undefined && (
                <div>
                  <span className="text-muted-foreground">Reconstruction Error:</span>
                  <span className="ml-2 font-mono">{anomalyResult.reconstructionError.toFixed(4)}</span>
                </div>
              )}
            </div>
          </div>

          {/* Stage 2: Attack Classification */}
          <div className={`border-l-4 pl-4 ${anomalyResult.isAnomaly ? 'border-orange-500' : 'border-gray-300'}`}>
            <div className="flex items-center mb-2">
              <h4 className="font-medium text-sm">Stage 2: Attack Classification</h4>
              {anomalyResult.isAnomaly && anomalyResult.attackType && anomalyResult.attackType.id !== undefined ? (
                <Shield className="h-4 w-4 text-orange-500 ml-2" />
              ) : (
                <CheckCircle className="h-4 w-4 text-gray-400 ml-2" />
              )}
            </div>
            {anomalyResult.isAnomaly && anomalyResult.attackType && anomalyResult.attackType.id !== undefined ? (
              <div className="space-y-2">
                <div className="text-sm">
                  <span className="text-muted-foreground">Attack Type:</span>
                  <span className="ml-2 font-medium" style={{ color: anomalyResult.attackType.color }}>
                    {anomalyResult.attackType.name}
                  </span>
                  <span className="text-xs text-muted-foreground ml-2">
                    ({anomalyResult.attackType.description})
                  </span>
                </div>
                {anomalyResult.attackConfidence !== undefined && (
                  <div className="text-sm">
                    <span className="text-muted-foreground">Classification Confidence:</span>
                    <span className="ml-2">
                      <div className="w-24 bg-secondary rounded-full h-2 inline-block align-middle">
                        <div
                          className="bg-orange-500 h-2 rounded-full"
                          style={{ width: `${anomalyResult.attackConfidence * 100}%` }}
                        />
                      </div>
                      <span className="ml-2 text-xs">
                        {(anomalyResult.attackConfidence * 100).toFixed(1)}%
                      </span>
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">
                {anomalyResult.isAnomaly ? 'Attack classification not available' : 'No classification needed'}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TwoStagePanel;
