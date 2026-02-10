import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, AlertTriangle, ArrowRight, Clock, CheckCircle2 } from 'lucide-react';
import Navbar from '../components/Layout/Navbar';
import Sidebar from '../components/Layout/Sidebar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { anomaliesApi, AnomalyData } from '../api/api';

const XAI: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [anomalies, setAnomalies] = useState<AnomalyData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchAnomalies = async () => {
      try {
        const data = await anomaliesApi.getAnomalies(1, 10);
        setAnomalies(data.data);
      } catch (error) {
        console.error('Failed to fetch anomalies:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnomalies();
  }, []);

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const getSeverityColor = (severity: AnomalyData['severity']) => {
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

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
          <div className="flex items-center mb-6">
            <BookOpen className="h-6 w-6 mr-2 text-cyberpurple" />
            <h1 className="text-2xl font-bold">XAI Explanations</h1>
          </div>


          <div className="grid gap-6">
            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Total Anomalies</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{anomalies.length}</div>
                  <p className="text-xs text-muted-foreground">
                    Available for explanation
                  </p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Explanations Generated</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{anomalies.length}</div>
                  <p className="text-xs text-muted-foreground">
                    Mock explanations available
                  </p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Model Types</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">3</div>
                  <p className="text-xs text-muted-foreground">
                    SHAP, LIME, Autoencoder
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Anomalies List */}
            <Card>
              <CardHeader>
                <CardTitle>Anomalies with XAI Explanations</CardTitle>
                <CardDescription>
                  Click on any anomaly to view its detailed XAI explanation
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="space-y-4">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <div key={i} className="h-20 bg-muted/30 animate-pulse rounded-md" />
                    ))}
                  </div>
                ) : anomalies.length > 0 ? (
                  <div className="space-y-4">
                    {anomalies.map((anomaly) => (
                      <div
                        key={anomaly.id}
                        className="flex items-center justify-between p-4 border rounded-lg hover:bg-secondary/50 transition-colors"
                      >
                        <div className="flex items-center space-x-4">
                          <div className="flex-shrink-0">
                            <AlertTriangle className="h-5 w-5 text-amber-500" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-2 mb-1">
                              <Badge className={getSeverityColor(anomaly.severity)} variant="outline">
                                {anomaly.severity.charAt(0).toUpperCase() + anomaly.severity.slice(1)}
                              </Badge>
                              <span className="text-sm text-muted-foreground">
                                {anomaly.sourceIp} → {anomaly.destinationIp}
                              </span>
                            </div>
                            <p className="text-sm text-muted-foreground truncate">
                              {anomaly.details}
                            </p>
                            <div className="flex items-center space-x-4 mt-1">
                              <span className="text-xs text-muted-foreground">
                                {formatDate(anomaly.timestamp)}
                              </span>
                              <div className="flex items-center">
                                {anomaly.reviewed ? (
                                  <CheckCircle2 className="h-3 w-3 text-green-500 mr-1" />
                                ) : (
                                  <Clock className="h-3 w-3 text-amber-500 mr-1" />
                                )}
                                <span className="text-xs text-muted-foreground">
                                  {anomaly.reviewed ? 'Reviewed' : 'Pending'}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                        <Button asChild variant="outline" size="sm">
                          <Link to={`/explanations/${anomaly.id}`}>
                            View Explanation
                            <ArrowRight className="h-4 w-4 ml-2" />
                          </Link>
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <BookOpen className="h-12 w-12 mx-auto mb-4" />
                    <p>No anomalies available for explanation</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* XAI Information */}
            <Card>
              <CardHeader>
                <CardTitle>About XAI Explanations</CardTitle>
                <CardDescription>
                  Understanding how our federated anomaly detection system makes decisions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-medium mb-2">SHAP (SHapley Additive exPlanations)</h3>
                    <p className="text-sm text-muted-foreground">
                      Provides feature importance scores by analyzing how each feature contributes 
                      to the anomaly detection decision across different model combinations.
                    </p>
                  </div>
                  <div>
                    <h3 className="font-medium mb-2">LIME (Local Interpretable Model-agnostic Explanations)</h3>
                    <p className="text-sm text-muted-foreground">
                      Generates local explanations by creating simplified models around specific 
                      predictions to understand feature contributions.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

export default XAI;
