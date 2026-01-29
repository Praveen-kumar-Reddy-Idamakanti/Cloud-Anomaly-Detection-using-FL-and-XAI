import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { EnhancedDetectionForm } from '../components/Detection/EnhancedDetectionForm';
import { EnhancedDetectionResults } from '../components/Detection/EnhancedDetectionResults';
import { AttackTypeVisualization } from '../components/Detection/AttackTypeVisualization';
import { RealtimeMonitor } from '../components/Detection/RealtimeMonitor';
import { XAIIntegrationPanel } from '../components/XAI/XAIIntegrationPanel';
import { DetectionResult } from '../api/api';
import { modelApi } from '../api/api';
import { Brain } from 'lucide-react';

export const DetectionPage: React.FC = () => {
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedResult, setSelectedResult] = useState<DetectionResult | null>(null);

  React.useEffect(() => {
    // Load model information on component mount
    const loadModelInfo = async () => {
      try {
        const info = await modelApi.getModelInfo();
        setModelInfo(info);
      } catch (error) {
        console.error('Failed to load model info:', error);
      }
    };
    
    loadModelInfo();
  }, []);

  const handleDetectionResults = (results: DetectionResult[]) => {
    setDetectionResults(prev => [...results, ...prev]);
    // Select the first result for XAI explanation
    if (results.length > 0) {
      setSelectedResult(results[0]);
    }
  };

  const handleRealtimeDetection = (result: DetectionResult) => {
    setDetectionResults(prev => [result, ...prev]);
    // Update selected result for XAI explanation
    setSelectedResult(result);
  };

  const clearResults = () => {
    setDetectionResults([]);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Anomaly Detection</h1>
          <p className="text-gray-600 mt-2">
            Advanced two-stage anomaly detection with attack type classification
          </p>
        </div>
        
        {modelInfo && (
          <Card className="w-80">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Model Status</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Status:</span>
                  <span className={modelInfo.status === 'loaded' ? 'text-green-600' : 'text-red-600'}>
                    {modelInfo.status}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Input Dimensions:</span>
                  <span>{modelInfo.input_dim}</span>
                </div>
                <div className="flex justify-between">
                  <span>Two-Stage:</span>
                  <span className={modelInfo.two_stage_enabled ? 'text-green-600' : 'text-gray-600'}>
                    {modelInfo.two_stage_enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
                {modelInfo.two_stage_enabled && (
                  <div className="flex justify-between">
                    <span>Attack Types:</span>
                    <span>{modelInfo.attack_types?.length || 0}</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Main Content */}
      <Tabs defaultValue="detect" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="detect">Detect</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="xai">XAI Explain</TabsTrigger>
          <TabsTrigger value="monitor">Real-time</TabsTrigger>
        </TabsList>

        {/* Detection Tab */}
        <TabsContent value="detect">
          <EnhancedDetectionForm onResults={handleDetectionResults} />
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-semibold">Detection Results</h2>
              {detectionResults.length > 0 && (
                <button
                  onClick={clearResults}
                  className="px-4 py-2 text-sm bg-red-500 text-white rounded hover:bg-red-600"
                >
                  Clear All Results
                </button>
              )}
            </div>
            
            <EnhancedDetectionResults results={detectionResults} />
          </div>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis">
          <div className="space-y-4">
            <h2 className="text-2xl font-semibold">Attack Type Analysis</h2>
            <AttackTypeVisualization results={detectionResults} />
          </div>
        </TabsContent>

        {/* XAI Explain Tab */}
        <TabsContent value="xai">
          <div className="space-y-4">
            <h2 className="text-2xl font-semibold">XAI Explainability</h2>
            {selectedResult ? (
              <XAIIntegrationPanel
                features={selectedResult.features}
                anomalyScore={selectedResult.anomalyScore}
                reconstructionError={selectedResult.anomalyScore}
                attackType={selectedResult.attackType?.id}
                confidence={selectedResult.attackConfidence}
              />
            ) : (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center py-8 text-gray-500">
                    <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No detection result selected for explanation</p>
                    <p className="text-sm">Run a detection to enable XAI explanations</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* Real-time Monitoring Tab */}
        <TabsContent value="monitor">
          <div className="space-y-4">
            <h2 className="text-2xl font-semibold">Real-time Monitoring</h2>
            <RealtimeMonitor onDetectionUpdate={handleRealtimeDetection} />
          </div>
        </TabsContent>
      </Tabs>

      {/* Summary Statistics */}
      {detectionResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Session Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{detectionResults.length}</div>
                <div className="text-sm text-gray-600">Total Analyses</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {detectionResults.filter(r => r.isAnomaly).length}
                </div>
                <div className="text-sm text-gray-600">Anomalies Detected</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {detectionResults.filter(r => !r.isAnomaly).length}
                </div>
                <div className="text-sm text-gray-600">Normal Traffic</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {detectionResults.filter(r => r.attackType).length}
                </div>
                <div className="text-sm text-gray-600">Attack Types Identified</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
