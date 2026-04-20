
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { AlertTriangle, ArrowLeft, BookOpen, CheckCircle2, Clock } from 'lucide-react';
import { toast } from 'sonner';
import Navbar from '../components/Layout/Navbar';
import Sidebar from '../components/Layout/Sidebar';
import ExplanationView from '../components/XAI/ExplanationView';
import AttackTypeBadge from '../components/AttackTypeBadge';
import TwoStagePanel from '../components/TwoStagePanel';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { anomaliesApi, explanationsApi, AnomalyData, ExplanationData, TwoStageDetectionResult, getAttackTypeInfo } from '../api/api';

const XAIExplanation: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [anomaly, setAnomaly] = useState<AnomalyData | null>(null);
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      if (!id) {
        console.log('=== XAI EXPLANATION PAGE: No ID provided ===');
        return;
      }
      
      console.log('=== XAI EXPLANATION PAGE: Fetching data for ID ===', id);
      setIsLoading(true);
      
      try {
        // Fetch anomaly details
        console.log('=== FETCHING ANOMALY DETAILS ===');
        const anomalyData = await anomaliesApi.getAnomalyById(id);
        console.log('Anomaly data received:', anomalyData);
        console.log('anomalyData.anomalyScore:', anomalyData.anomalyScore);
        console.log('anomalyData.reconstructionError:', anomalyData.reconstructionError);
        console.log('anomalyData.attackType:', anomalyData.attackType);
        console.log('anomalyData.attackConfidence:', anomalyData.attackConfidence);
        setAnomaly(anomalyData);
        
        // Extract features from anomaly details
        let features: number[] | null = null;
        try {
          // Features are stored in the 'features' field as JSON string
          if (anomalyData.features) {
            console.log('Parsing features from anomaly data...');
            features = JSON.parse(anomalyData.features);
            console.log('Features parsed successfully:', features.length, 'features');
          } else {
            console.log('No features field in anomaly data');
          }
        } catch (parseError) {
          console.error("Failed to parse anomaly features:", parseError);
        }

        if (!features) {
          console.error('No features available for explanation');
          toast.error('Anomaly features not found for explanation.');
          setExplanation(null);
          setIsLoading(false);
          return;
        }

        console.log('=== CALLING EXPLANATION API ===');
        console.log('Features length:', features.length);
        console.log('Features sample:', features.slice(0, 5));

        // Use mock explanations directly
        try {
          const { getMockAnomalyById } = await import('../data/mockAnomalyExplanations');
          const mockData = getMockAnomalyById(id || '');
          if (mockData) {
            console.log("=== USING MOCK EXPLANATION DATA ===");
            console.log('Mock explanation data:', mockData.explanation);
            console.log('Mock attack type:', mockData.explanation.phase3?.attack_name);
            console.log('Mock confidence:', mockData.explanation.phase3?.confidence);
            setExplanation(mockData.explanation);
          } else {
            console.log("=== NO MOCK DATA FOUND, FALLING BACK TO API ===");
            // Fallback to real API if no mock data found
            const explanationData = await explanationsApi.getAnomalyExplanation(features);
            console.log('=== EXPLANATION DATA RECEIVED ===');
            console.log('Explanation data:', explanationData);
            setExplanation(explanationData);
          }
        } catch (mockError) {
          console.error("Failed to load mock explanation, falling back to API:", mockError);
          // Final fallback to real API
          try {
            const explanationData = await explanationsApi.getAnomalyExplanation(features);
            setExplanation(explanationData);
          } catch (apiError) {
            console.error("API also failed:", apiError);
            setExplanation(null);
          }
        }
      } catch (error: any) {
        console.error('=== ANOMALY FETCH ERROR ===');
        console.error("Failed to fetch anomaly data:", error);
        toast.error(error.message || 'Failed to fetch anomaly data');
        setAnomaly(null);
      } finally {
        console.log('=== FETCH COMPLETED ===');
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, [id]);

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const getSeverityColor = (severity?: AnomalyData['severity']) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'high':
        return 'bg-amber-500/10 text-amber-500 border-amber-500/20';
      case 'medium':
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      case 'low':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      default:
        return 'bg-secondary text-secondary-foreground';
    }
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const handleMarkAsReviewed = async () => {
    if (!anomaly) return;
    
    try {
      const updatedAnomaly = await anomaliesApi.reviewAnomaly(anomaly.id, true);
      setAnomaly(updatedAnomaly);
      toast.success('Anomaly marked as reviewed');
    } catch (error: any) {
      toast.error(error.message || 'Failed to update anomaly status');
    }
  };

  const anomalyResultForPanel = anomaly && explanation ? {
    id: anomaly.id,
    timestamp: new Date(anomaly.timestamp),
    features: anomaly.features ? JSON.parse(anomaly.features) : [],
    isAnomaly: explanation.anomaly_detected ?? (anomaly.severity !== 'low'),
    anomalyScore: explanation.reconstruction_error ?? anomaly.anomalyScore ?? 0,
    threshold: 0.22610116, // Default threshold
    reconstructionError: explanation.reconstruction_error ?? anomaly.reconstructionError,
    attackType: anomaly.attackType || (explanation.phase3?.attack_type !== undefined && explanation.phase3?.attack_type !== null ? getAttackTypeInfo(explanation.phase3.attack_type) : undefined),
    attackConfidence: explanation.phase3?.confidence ?? anomaly.attackConfidence,
    confidence: anomaly.confidence
  } : null;

  console.log('anomalyResultForPanel before passing to TwoStagePanel:', anomalyResultForPanel);
  console.log('anomaly.attackType:', anomaly?.attackType);
  console.log('explanation.phase3?.attack_type:', explanation?.phase3?.attack_type);

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
          <div className="mb-6">
            <Button asChild variant="outline" size="sm" className="mb-4">
              <Link to="/anomalies">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Anomalies
              </Link>
            </Button>
            
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-center">
                <BookOpen className="h-6 w-6 mr-2 text-cyberpurple" />
                <h1 className="text-2xl font-bold">XAI Explanation</h1>
              </div>
              
              {anomaly && !anomaly.reviewed && (
                <Button onClick={handleMarkAsReviewed}>
                  <CheckCircle2 className="h-4 w-4 mr-2" />
                  Mark as Reviewed
                </Button>
              )}
            </div>
          </div>

          {isLoading ? (
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <div className="h-7 bg-muted/30 animate-pulse rounded-md w-1/3 mb-2" />
                  <div className="h-5 bg-muted/30 animate-pulse rounded-md w-1/2" />
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Array.from({ length: 4 }).map((_, i) => (
                      <div key={i} className="h-20 bg-muted/30 animate-pulse rounded-md" />
                    ))}
                  </div>
                </CardContent>
              </Card>
              
              <ExplanationView explanation={null} isLoading={true} />
            </div>
          ) : anomaly ? (
            <div className="space-y-6">
              {/* Anomaly Details */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <AlertTriangle className="mr-2 h-5 w-5 text-amber-500" />
                    Anomaly Details
                  </CardTitle>
                  <CardDescription>
                    Detection information and network details for this security event
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Severity</p>
                      <Badge className={getSeverityColor(anomaly.severity)} variant="outline">
                        {anomaly.severity.charAt(0).toUpperCase() + anomaly.severity.slice(1)}
                      </Badge>
                    </div>
                    
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Status</p>
                      <div className="flex items-center">
                        {anomaly.reviewed ? (
                          <CheckCircle2 className="h-4 w-4 text-green-500 mr-1" />
                        ) : (
                          <Clock className="h-4 w-4 text-amber-500 mr-1" />
                        )}
                        <span>{anomaly.reviewed ? 'Reviewed' : 'Pending Review'}</span>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Confidence</p>
                      <div>
                        <div className="w-full bg-secondary rounded-full h-2.5">
                          <div
                            className="bg-cyberpurple h-2.5 rounded-full"
                            style={{ width: `${anomaly.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span>{(anomaly.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Timestamp</p>
                      <p className="font-mono text-sm">{formatDate(anomaly.timestamp)}</p>
                    </div>
                  </div>
                  
                  <Separator className="my-4" />
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium">Network Information</h3>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-sm text-muted-foreground">Source IP</p>
                          <p className="font-mono text-sm">{anomaly.sourceIp}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Destination IP</p>
                          <p className="font-mono text-sm">{anomaly.destinationIp}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Protocol</p>
                          <p className="text-sm">{anomaly.protocol}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Action</p>
                          <p className="text-sm">{anomaly.action}</p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium">Details</h3>
                      <p>{anomaly.details}</p>
                      
                      {/* Attack Type Information */}
                      {anomaly.attackType && (
                        <div className="mt-3">
                          <p className="text-sm text-muted-foreground">Attack Type</p>
                          <AttackTypeBadge 
                            attackType={anomaly.attackType} 
                            confidence={anomaly.attackConfidence} 
                          />
                        </div>
                      )}
                      
                      {explanation && (
                        <div>
                          <p className="text-sm text-muted-foreground">Model Type</p>
                          <p className="text-sm">{explanation.model_type}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              {/* Two-Stage Detection Panel */}
              {anomaly && anomalyResultForPanel && (
                <TwoStagePanel anomalyResult={anomalyResultForPanel} />
              )}
              
              {/* XAI Explanation */}
              <ExplanationView explanation={explanation} />
            </div>
          ) : (
            <div className="text-center py-12">
              <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h2 className="text-xl font-medium mb-2">Anomaly Not Found</h2>
              <p className="text-muted-foreground mb-6">
                We couldn't find this anomaly in our system.
              </p>
              <Button asChild variant="outline">
                <Link to="/anomalies">Back to Anomalies</Link>
              </Button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default XAIExplanation;
