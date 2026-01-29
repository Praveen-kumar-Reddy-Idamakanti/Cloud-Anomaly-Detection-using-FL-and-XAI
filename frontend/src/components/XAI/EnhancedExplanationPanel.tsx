import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import { Brain, Target, Zap, RefreshCw, AlertTriangle, TrendingUp } from 'lucide-react';
import { explanationsApi } from '../../api/api';

interface EnhancedExplanationPanelProps {
  features: number[];
  anomalyScore?: number;
  reconstructionError?: number;
  attackType?: number;
  confidence?: number;
  onPhaseChange?: (phase: string) => void;
}

export const EnhancedExplanationPanel: React.FC<EnhancedExplanationPanelProps> = ({
  features,
  anomalyScore = 0.3,
  reconstructionError = 0.3,
  attackType,
  confidence = 0.8,
  onPhaseChange
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
      onPhaseChange?.(phase);
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

  const renderPhaseContent = () => {
    if (!explanation) return null;

    switch (selectedPhase) {
      case 'phase1':
        return (
          <div className="space-y-4">
            <div className="flex items-center space-x-2 mb-4">
              {getPhaseIcon('phase1')}
              <h3 className="text-lg font-semibold">Phase 1: Foundation Setup</h3>
            </div>
            
            <Alert>
              <Brain className="h-4 w-4" />
              <AlertDescription>
                <strong>Basic Anomaly Detection Explanation</strong>
                <p className="mt-2">
                  This phase provides fundamental insights into why the model classified the data as an anomaly.
                </p>
              </AlertDescription>
            </Alert>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-sm mb-2">Detection Result</h4>
                <div className="text-sm text-gray-600">
                  {explanation.phase1?.is_anomaly ? 'Anomaly Detected' : 'Normal Traffic'}
                </div>
                <div className="text-sm text-gray-500">
                  Confidence: {((explanation.phase1?.confidence || 0) * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <h4 className="font-medium text-sm mb-2">Reasoning</h4>
                <div className="text-sm text-gray-600">
                  {explanation.phase1?.reasoning || 'No reasoning available'}
                </div>
                <div className="text-sm text-gray-500">
                  Key Features: {explanation.phase1?.key_features?.length || 0}
                </div>
              </div>
            </div>
          </div>
        );
      
      case 'phase2':
        return (
          <div className="space-y-4">
            <div className="flex items-center space-x-2 mb-4">
              {getPhaseIcon('phase2')}
              <h3 className="text-lg font-semibold">Phase 2: Autoencoder Explainability</h3>
            </div>
            
            <Alert>
              <Target className="h-4 w-4" />
              <AlertDescription>
                <strong>SHAP-based Feature Importance</strong>
                <p className="mt-2">
                  This phase uses SHAP values to explain which features contributed most to the anomaly detection.
                </p>
              </AlertDescription>
            </Alert>
            
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-sm mb-2">Feature Importance</h4>
                <div className="space-y-2">
                  {explanation.phase2?.feature_importance?.slice(0, 5).map((feature: any, index: number) => (
                    <div key={index} className="flex justify-between items-center p-2 border rounded">
                      <span className="text-sm font-mono">{feature.feature_name}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500">SHAP:</span>
                        <span className={`text-xs font-mono ${feature.direction === 'positive' ? 'text-green-600' : 'text-red-600'}`}>
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
              
              <div>
                <h4 className="font-medium text-sm mb-2">Reconstruction Analysis</h4>
                <div className="text-sm text-gray-600">
                  Reconstruction Error: {explanation.phase2?.reconstruction_error?.toFixed(4) || 'N/A'}
                </div>
                <div className="text-sm text-gray-500">
                  Threshold: {0.2261}
                </div>
              </div>
            </div>
          </div>
        );
      
      case 'phase3':
        return (
          <div className="space-y-4">
            <div className="flex items-center space-x-2 mb-4">
              {getPhaseIcon('phase3')}
              <h3 className="text-lg font-semibold">Phase 3: Attack Type Explainability</h3>
            </div>
            
            <Alert>
              <Zap className="h-4 w-4" />
              <AlertDescription>
                <strong>Attack Type Classification Explainability</strong>
                <p className="mt-2">
                  This phase explains why the model classified the anomaly as a specific attack type.
                </p>
              </AlertDescription>
            </Alert>
            
            {explanation.phase3?.explanation_type === 'attack_type_explainability' ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-sm mb-2">Attack Classification</h4>
                    <div className="text-sm text-gray-600">
                      Predicted: {explanation.phase3?.attack_name || 'Unknown'}
                    </div>
                    <div className="text-sm text-gray-500">
                      Confidence: {((explanation.phase3?.confidence || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium text-sm mb-2">Classification Reasoning</h4>
                    <div className="text-sm text-gray-600">
                      {explanation.phase3?.explanation?.confidence_reasoning || 'No reasoning available'}
                    </div>
                    <div className="text-sm text-gray-500">
                      Key Indicators: {explanation.phase3?.explanation?.key_indicators?.length || 0}
                    </div>
                  </div>
                </div>
                
                <div className="mt-4 p-4 border rounded-lg">
                  <h4 className="font-medium text-sm mb-2">Attack Type Details</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Attack Type:</span>
                      <Badge variant="outline" style={{ 
                        borderColor: '#ef4444', 
                        color: '#ef4444' 
                      }}>
                        {explanation.phase3?.attack_name || 'Unknown'}
                      </Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Confidence:</span>
                      <span className="text-gray-600">
                        {((explanation.phase3?.confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No attack type classification available</p>
              </div>
            )}
          </div>
        );
      
      default:
        return (
          <div className="space-y-4">
            <div className="flex items-center space-x-2 mb-4">
              {getPhaseIcon('comprehensive')}
              <h3 className="text-lg font-semibold">Comprehensive Explanation</h3>
            </div>
            
            <Alert>
              <TrendingUp className="h-4 w-4" />
              <AlertDescription>
                <strong>All Phases Combined</strong>
                <p className="mt-2">
                  This comprehensive explanation combines all three XAI phases for complete transparency.
                </p>
              </AlertDescription>
            </Alert>
            
            {explanation && (
              <div className="space-y-6">
                {/* Phase 1 Summary */}
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    {getPhaseIcon('phase1')}
                    <h4 className="font-medium">Phase 1 Summary</h4>
                  </div>
                  <div className="text-sm text-gray-600">
                    <strong>Result:</strong> {explanation.phase1?.is_anomaly ? 'Anomaly Detected' : 'Normal Traffic'}
                    <br />
                    <strong>Confidence:</strong> {((explanation.phase1?.confidence || 0) * 100).toFixed(1)}%
                  </div>
                </div>
                
                {/* Phase 2 Summary */}
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    {getPhaseIcon('phase2')}
                    <h4 className="font-medium">Phase 2 Summary</h4>
                  </div>
                  <div className="text-sm text-gray-600">
                    <strong>Top Features:</strong> {explanation.phase2?.feature_importance?.slice(0, 3).map(f => f.feature_name).join(', ') || 'N/A'}
                    <br />
                    <strong>SHAP Range:</strong> 
                    [{explanation.phase2?.shap_values?.slice(0, 3).map(v => v.toFixed(4)).join(', ')}, 
                     {explanation.phase2?.shap_values?.slice(-3).map(v => v.toFixed(4)).join(', ')}]
                  </div>
                </div>
                
                {/* Phase 3 Summary */}
                {explanation.phase3?.explanation_type === 'attack_type_explainability' ? (
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      {getPhaseIcon('phase3')}
                      <h4 className="font-medium">Phase 3 Summary</h4>
                    </div>
                    <div className="text-sm text-gray-600">
                      <strong>Attack Type:</strong> {explanation.phase3?.attack_name || 'Unknown'}
                      <br />
                      <strong>Confidence:</strong> {((explanation.phase3?.confidence || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No attack classification available</p>
                  </div>
                )}
                
                {/* Overall Metrics */}
                <div className="p-4 border rounded-lg bg-gray-50">
                  <h4 className="font-medium mb-2">Overall Assessment</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <span>Model Performance:</span>
                      <span className="text-gray-600">
                        {explanation.anomaly_detected ? 'Anomaly detected' : 'Normal traffic'}
                      </span>
                    </div>
                    <div>
                      <span>Explanation Depth:</span>
                      <span className="text-gray-600">
                        {Object.keys(explanation).length} phases available
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center">
            <Zap className="h-5 w-5 mr-2" />
            Enhanced XAI Explanation Panel
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
        {renderPhaseContent()}
        
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
