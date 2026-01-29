import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import { Brain, Target, Zap, RefreshCw } from 'lucide-react';
import { explanationsApi } from '../../api/api';

interface XAIIntegrationPanelProps {
  features: number[];
  anomalyScore?: number;
  reconstructionError?: number;
  attackType?: number;
  confidence?: number;
}

export const XAIIntegrationPanel: React.FC<XAIIntegrationPanelProps> = ({
  features,
  anomalyScore = 0.3,
  reconstructionError = 0.3,
  attackType,
  confidence = 0.8
}) => {
  const [selectedPhase, setSelectedPhase] = useState('comprehensive');
  const [explanation, setExplanation] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePhaseChange = async (phase: string) => {
    setSelectedPhase(phase);
    setIsLoading(true);
    try {
      let result;
      
      switch (phase) {
        case 'phase1':
          result = await explanationsApi.getPhaseExplanation('phase1', features, { anomaly_score: anomalyScore });
          break;
        case 'phase2':
          result = await explanationsApi.getPhaseExplanation('phase2', features, { reconstruction_error: reconstructionError });
          break;
        case 'phase3':
          if (attackType !== undefined) {
            result = await explanationsApi.getAttackTypeExplanation(features, attackType, confidence);
          } else {
            result = { phase: 'phase3_classification', explanation_type: 'not_anomaly' };
          }
          break;
        default:
          result = await explanationsApi.getComprehensiveExplanation(features);
          break;
      }
      
      setExplanation(result);
    } catch (error) {
      console.error(`Failed to get ${phase} explanation:`, error);
      setExplanation(null);
    } finally {
      setIsLoading(false);
    }
  };

  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case 'phase1': return <Brain className="h-4 w-4 text-blue-500" />;
      case 'phase2': return <Target className="h-4 w-4 text-purple-500" />;
      case 'phase3': return <Zap className="h-4 w-4 text-orange-500" />;
      default: return <Brain className="h-4 w-4 text-gray-500" />;
    }
  };

  const getPhaseDescription = (phase: string) => {
    switch (phase) {
      case 'phase1': return "Phase 1: Foundation Setup - Basic anomaly detection explanation";
      case 'phase2': return "Phase 2: Autoencoder Explainability - SHAP-based feature importance";
      case 'phase3': return "Phase 3: Attack Type Classification Explainability - Attack type reasoning";
      default: return "Comprehensive explanation - All phases combined";
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center">
            <Zap className="h-5 w-5 mr-2" />
            XAI Integration Panel
          </div>
          <Badge variant="outline">
            {explanation ? 'Ready' : 'No Data'}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Phase Selection */}
        <div className="flex items-center space-x-2 mb-6">
          {['comprehensive', 'phase1', 'phase2', 'phase3'].map((phase) => (
            <Button
              key={phase}
              variant={selectedPhase === phase ? "default" : "outline"}
              size="sm"
              onClick={() => handlePhaseChange(phase)}
              className={selectedPhase === phase ? "ring-2 ring-blue-500" : ""}
            >
              {getPhaseIcon(phase)}
              <span className="ml-2">{phase.charAt(0).toUpperCase() + phase.slice(1)}</span>
            </Button>
          ))}
        </div>
        
        {/* Phase Description */}
        <div className="mb-4">
          <p className="text-sm text-gray-600">
            {getPhaseDescription(selectedPhase)}
          </p>
        </div>
        
        {/* Phase Content */}
        {explanation && (
          <div className="space-y-4">
            <Alert>
              <Brain className="h-4 w-4" />
              <AlertDescription>
                <strong>XAI Explanation Available</strong>
                <p className="mt-2">
                  Successfully generated {selectedPhase} explanation for the provided features.
                </p>
              </AlertDescription>
            </Alert>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-sm mb-2">Phase Information</h4>
                <div className="text-sm text-gray-600">
                  Phase: {explanation.phase || selectedPhase}
                </div>
                <div className="text-sm text-gray-600">
                  Type: {explanation.explanation_type || 'unknown'}
                </div>
                <div className="text-sm text-gray-600">
                  Features: {explanation.features?.length || 0}
                </div>
              </div>
              <div>
                <h4 className="font-medium text-sm mb-2">Detection Results</h4>
                <div className="text-sm text-gray-600">
                  Anomaly Detected: {explanation.anomaly_detected ? 'Yes' : 'No'}
                </div>
                <div className="text-sm text-gray-600">
                  Reconstruction Error: {explanation.reconstruction_error?.toFixed(4) || 'N/A'}
                </div>
                <div className="text-sm text-gray-600">
                  Timestamp: {explanation.timestamp || 'N/A'}
                </div>
              </div>
            </div>
            
            {/* Phase-specific content */}
            {selectedPhase === 'phase2' && explanation.feature_importance && (
              <div>
                <h4 className="font-medium text-sm mb-2">Feature Importance</h4>
                <div className="space-y-2">
                  {explanation.feature_importance.slice(0, 5).map((feature: any, index: number) => (
                    <div key={index} className="flex justify-between items-center p-2 border rounded">
                      <span className="text-sm font-mono">{feature.feature_name}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500">SHAP:</span>
                        <span className="text-xs font-mono">
                          {feature.shap_value.toFixed(4)}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500">
                        {(feature.importance * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {selectedPhase === 'phase3' && explanation.attack_name && (
              <div>
                <h4 className="font-medium text-sm mb-2">Attack Type Information</h4>
                <div className="p-4 border rounded-lg">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Attack Type:</span>
                      <Badge variant="outline" style={{ 
                        borderColor: '#ef4444', 
                        color: '#ef4444' 
                      }}>
                        {explanation.attack_name}
                      </Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Confidence:</span>
                      <span className="text-gray-600">
                        {((explanation.confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <span className="ml-2">Generating explanation...</span>
          </div>
        )}
        
        {/* No Data State */}
        {!explanation && !isLoading && (
          <div className="text-center py-8 text-gray-500">
            <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No explanation available</p>
            <p className="text-sm">Select a phase to generate explanation</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
