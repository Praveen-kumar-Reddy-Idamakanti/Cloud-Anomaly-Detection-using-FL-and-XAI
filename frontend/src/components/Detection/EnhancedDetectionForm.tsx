import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Switch } from '../ui/switch';
import { Alert, AlertDescription } from '../ui/alert';
import { modelApi, EnhancedDetectionRequest, DetectionResult } from '../../api/api';
import { EnhancedDetectionResults } from './EnhancedDetectionResults';

interface EnhancedDetectionFormProps {
  onResults?: (results: DetectionResult[]) => void;
}

export const EnhancedDetectionForm: React.FC<EnhancedDetectionFormProps> = ({
  onResults
}) => {
  const [features, setFeatures] = useState<string>('');
  const [threshold, setThreshold] = useState<string>('0.22610116');
  const [useEnhanced, setUseEnhanced] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [results, setResults] = useState<DetectionResult[]>([]);
  const [error, setError] = useState<string>('');

  const validateFeatures = (featureString: string): number[][] | null => {
    try {
      const lines = featureString.trim().split('\\n').filter(line => line.trim());
      const featureArrays: number[][] = [];
      
      for (const line of lines) {
        const values = line.split(',').map(v => parseFloat(v.trim()));
        if (values.some(isNaN)) {
          throw new Error('Invalid numeric values');
        }
        if (values.length !== 78) {
          throw new Error(`Expected 78 features, got ${values.length}`);
        }
        featureArrays.push(values);
      }
      
      return featureArrays;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Invalid feature format');
      return null;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    const featureArrays = validateFeatures(features);
    if (!featureArrays) return;

    setIsLoading(true);
    
    try {
      let response;
      
      if (useEnhanced) {
        response = await modelApi.detectAnomaliesEnhanced(featureArrays, parseFloat(threshold));
        // Convert enhanced response to DetectionResult format
        const detectionResults = featureArrays.map((features, index) => ({
          id: `detection_${Date.now()}_${index}`,
          timestamp: new Date(),
          features,
          isAnomaly: response.anomaly_predictions[index] === 1,
          anomalyScore: response.reconstruction_errors[index],
          threshold: response.threshold,
          attackType: response.anomaly_predictions[index] === 1 
            ? {
                id: response.attack_type_predictions[index],
                name: response.attack_types[response.attack_type_predictions[index]],
                description: `Attack type ${response.attack_type_predictions[index]}`,
                severity: 'high' as const,
                color: '#ef4444'
              }
            : undefined,
          attackConfidence: response.anomaly_predictions[index] === 1 
            ? response.attack_confidences[index] 
            : undefined,
          confidence: 1 - (response.reconstruction_errors[index] / (response.threshold * 2))
        }));
        setResults(detectionResults);
        onResults?.(detectionResults);
      } else {
        response = await modelApi.detectAnomalies(featureArrays, parseFloat(threshold));
        // Convert standard response to DetectionResult format
        const detectionResults = featureArrays.map((features, index) => ({
          id: `detection_${Date.now()}_${index}`,
          timestamp: new Date(),
          features,
          isAnomaly: response.predictions[index] === 1,
          anomalyScore: response.scores[index],
          threshold: response.threshold,
          attackType: undefined,
          attackConfidence: undefined,
          confidence: response.confidence[index]
        }));
        setResults(detectionResults);
        onResults?.(detectionResults);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Detection failed');
    } finally {
      setIsLoading(false);
    }
  };

  const loadSampleData = () => {
    // Generate sample 78-feature data
    const sampleFeatures = Array.from({ length: 3 }, () =>
      Array.from({ length: 78 }, () => (Math.random() * 2 - 1).toFixed(6)).join(', ')
    ).join('\\n');
    
    setFeatures(sampleFeatures);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Enhanced Anomaly Detection
            <div className="flex items-center space-x-2">
              <Label htmlFor="enhanced-mode">Two-Stage Detection</Label>
              <Switch
                id="enhanced-mode"
                checked={useEnhanced}
                onCheckedChange={setUseEnhanced}
              />
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="features">Network Features (78 dimensions)</Label>
              <Textarea
                id="features"
                placeholder="Enter 78 comma-separated features per line. Example:&#10;0.1,0.2,0.3,...,0.78&#10;0.5,0.6,0.7,...,0.58"
                value={features}
                onChange={(e) => setFeatures(e.target.value)}
                rows={6}
                className="font-mono text-sm"
              />
              <div className="flex justify-between">
                <p className="text-sm text-gray-600">
                  Each line should contain 78 comma-separated numeric values
                </p>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={loadSampleData}
                >
                  Load Sample Data
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="threshold">Anomaly Threshold</Label>
              <Input
                id="threshold"
                type="number"
                step="0.001"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                placeholder="0.22610116"
              />
              <p className="text-sm text-gray-600">
                Default threshold for optimal detection performance
              </p>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Detection Mode: {useEnhanced ? 'Two-Stage Enhanced' : 'Standard'}</h4>
              <p className="text-sm text-gray-600">
                {useEnhanced 
                  ? 'Advanced detection with attack type classification using autoencoder + attack classifier'
                  : 'Standard anomaly detection using reconstruction error thresholding'
                }
              </p>
            </div>

            <Button 
              type="submit" 
              disabled={isLoading || !features.trim()}
              className="w-full"
            >
              {isLoading ? 'Analyzing...' : 'Detect Anomalies'}
            </Button>
          </form>
        </CardContent>
      </Card>

      {results.length > 0 && (
        <EnhancedDetectionResults results={results} isLoading={isLoading} />
      )}
    </div>
  );
};
