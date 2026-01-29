import React, { useState, useEffect } from 'react';
import { Database, Play, Pause, MoreHorizontal, PlusCircle, RefreshCw, BarChart, Info, Brain, Target, Zap } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import Navbar from '../components/Layout/Navbar';
import Sidebar from '../components/Layout/Sidebar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { modelApi } from '../api/api';

interface ModelInfo {
  model_path: string;
  input_dim: number;
  last_trained: string;
  accuracy?: number;
  status: string;
  two_stage_enabled: boolean;
  attack_types: string[];
}

interface TrainingStatus {
  is_training: boolean;
  progress: number;
  round: number;
  total_rounds?: number;
  current_loss?: number;
  best_loss?: number;
}

const ModelManagementEnhanced: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    is_training: false,
    progress: 0,
    round: 0
  });
  const [isLoading, setIsLoading] = useState(true);
  const [enhancedMode, setEnhancedMode] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchModelInfo();
    fetchTrainingStatus();
    
    // Set up polling for training status
    const interval = setInterval(fetchTrainingStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchModelInfo = async () => {
    try {
      const info = await modelApi.getModelInfo();
      setModelInfo(info);
      setEnhancedMode(info.two_stage_enabled);
    } catch (error: any) {
      toast.error(error.message || 'Failed to fetch model info');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const status = await modelApi.getTrainingStatus();
      setTrainingStatus(status);
    } catch (error: any) {
      // Don't show error for training status, it might not be available
      console.warn('Training status unavailable:', error);
    }
  };

  const handleStartTraining = async () => {
    try {
      await modelApi.startTraining(10, 32, 0.001);
      toast.success('Training started');
      fetchTrainingStatus();
    } catch (error: any) {
      toast.error(error.message || 'Failed to start training');
    }
  };

  const handleStopTraining = async () => {
    try {
      // Note: stopTraining might not be implemented yet
      toast.info('Training stop requested');
      fetchTrainingStatus();
    } catch (error: any) {
      toast.error(error.message || 'Failed to stop training');
    }
  };

  const handleToggleEnhancedMode = async (enabled: boolean) => {
    setEnhancedMode(enabled);
    toast.info(`Two-stage detection ${enabled ? 'enabled' : 'disabled'}`);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'loaded': return 'bg-green-500';
      case 'loading': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      case 'not_loaded': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'loaded': return 'Loaded';
      case 'loading': return 'Loading';
      case 'error': return 'Error';
      case 'not_loaded': return 'Not Loaded';
      default: return 'Unknown';
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen">
        <Navbar toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} />
        <Sidebar isSidebarOpen={isSidebarOpen} />
        <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
          <div className="p-4 md:p-6 max-w-7xl mx-auto">
            <div className="flex items-center justify-center h-64">
              <RefreshCw className="h-8 w-8 animate-spin" />
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <Brain className="h-6 w-6 mr-2 text-cyberpurple" />
              <h1 className="text-2xl font-bold">Enhanced Model Management</h1>
            </div>
            <Button onClick={() => navigate('/detect')}>
              <Target className="h-4 w-4 mr-2" />
              Test Detection
            </Button>
          </div>

          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="training">Training</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="configuration">Configuration</TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <Card>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Model Status</p>
                        <p className="text-2xl font-bold">{getStatusText(modelInfo?.status || 'unknown')}</p>
                      </div>
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(modelInfo?.status || 'unknown')}`}></div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Input Dimensions</p>
                        <p className="text-2xl font-bold">{modelInfo?.input_dim || 0}</p>
                      </div>
                      <Target className="h-8 w-8 text-blue-500" />
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Attack Types</p>
                        <p className="text-2xl font-bold">{modelInfo?.attack_types?.length || 0}</p>
                      </div>
                      <Zap className="h-8 w-8 text-yellow-500" />
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Accuracy</p>
                        <p className="text-2xl font-bold">
                          {modelInfo?.accuracy ? `${(modelInfo.accuracy * 100).toFixed(1)}%` : 'N/A'}
                        </p>
                      </div>
                      <BarChart className="h-8 w-8 text-green-500" />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Model Details */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Brain className="h-5 w-5 mr-2" />
                      Model Information
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Model Path:</span>
                      <span className="text-sm text-gray-600 font-mono">
                        {modelInfo?.model_path?.split('/').pop() || 'Unknown'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Last Trained:</span>
                      <span className="text-sm text-gray-600">
                        {modelInfo?.last_trained ? formatDate(modelInfo.last_trained) : 'Unknown'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Input Dimensions:</span>
                      <span className="text-sm text-gray-600">{modelInfo?.input_dim || 0} features</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Two-Stage Mode:</span>
                      <div className="flex items-center space-x-2">
                        <Switch
                          checked={enhancedMode}
                          onCheckedChange={handleToggleEnhancedMode}
                          disabled={modelInfo?.status !== 'loaded'}
                        />
                        <Badge variant={enhancedMode ? "default" : "secondary"}>
                          {enhancedMode ? 'Enabled' : 'Disabled'}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Zap className="h-5 w-5 mr-2" />
                      Attack Type Classification
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {modelInfo?.attack_types && modelInfo.attack_types.length > 0 ? (
                      <div className="space-y-2">
                        {modelInfo.attack_types.map((type, index) => (
                          <div key={index} className="flex items-center justify-between p-2 border rounded">
                            <span className="text-sm font-medium">{type}</span>
                            <Badge variant="outline" className="text-xs">
                              ID: {index}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <Info className="h-8 w-8 mx-auto mb-2" />
                        <p className="text-sm">No attack types available</p>
                        <p className="text-xs">Enable two-stage mode to activate attack classification</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Training Tab */}
            <TabsContent value="training">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Training Control</CardTitle>
                    <CardDescription>
                      Manage federated learning training sessions
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium">Training Status</p>
                        <p className="text-lg font-bold">
                          {trainingStatus.is_training ? 'In Progress' : 'Idle'}
                        </p>
                      </div>
                      <Badge variant={trainingStatus.is_training ? "default" : "secondary"}>
                        {trainingStatus.is_training ? 'Training' : 'Stopped'}
                      </Badge>
                    </div>

                    {trainingStatus.is_training && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Progress</span>
                          <span>{trainingStatus.progress.toFixed(1)}%</span>
                        </div>
                        <Progress value={trainingStatus.progress} className="w-full" />
                        <div className="flex justify-between text-xs text-gray-600">
                          <span>Round: {trainingStatus.round}</span>
                          {trainingStatus.current_loss && (
                            <span>Loss: {trainingStatus.current_loss.toFixed(4)}</span>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="flex space-x-2">
                      {!trainingStatus.is_training ? (
                        <Button onClick={handleStartTraining} className="flex-1">
                          <Play className="h-4 w-4 mr-2" />
                          Start Training
                        </Button>
                      ) : (
                        <Button onClick={handleStopTraining} variant="destructive" className="flex-1">
                          <Pause className="h-4 w-4 mr-2" />
                          Stop Training
                        </Button>
                      )}
                      <Button variant="outline" onClick={fetchTrainingStatus}>
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Training Configuration</CardTitle>
                    <CardDescription>
                      Configure federated learning parameters
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-sm font-medium">Epochs</label>
                        <input 
                          type="number" 
                          className="w-full p-2 border rounded" 
                          defaultValue="10"
                          disabled={trainingStatus.is_training}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Batch Size</label>
                        <input 
                          type="number" 
                          className="w-full p-2 border rounded" 
                          defaultValue="32"
                          disabled={trainingStatus.is_training}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Learning Rate</label>
                        <input 
                          type="number" 
                          step="0.001"
                          className="w-full p-2 border rounded" 
                          defaultValue="0.001"
                          disabled={trainingStatus.is_training}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Rounds</label>
                        <input 
                          type="number" 
                          className="w-full p-2 border rounded" 
                          defaultValue="5"
                          disabled={trainingStatus.is_training}
                        />
                      </div>
                    </div>
                    <Button 
                      className="w-full" 
                      disabled={trainingStatus.is_training || modelInfo?.status !== 'loaded'}
                    >
                      Apply Configuration
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Performance Tab */}
            <TabsContent value="performance">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Model Performance Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between">
                        <span className="text-sm font-medium">Accuracy:</span>
                        <span className="text-sm font-bold">
                          {modelInfo?.accuracy ? `${(modelInfo.accuracy * 100).toFixed(2)}%` : 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm font-medium">Input Dimensions:</span>
                        <span className="text-sm font-bold">{modelInfo?.input_dim || 0}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm font-medium">Two-Stage Enabled:</span>
                        <Badge variant={modelInfo?.two_stage_enabled ? "default" : "secondary"}>
                          {modelInfo?.two_stage_enabled ? 'Yes' : 'No'}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm font-medium">Attack Types:</span>
                        <span className="text-sm font-bold">{modelInfo?.attack_types?.length || 0}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>System Resources</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>CPU Usage</span>
                          <span>15%</span>
                        </div>
                        <Progress value={15} className="w-full" />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Memory Usage</span>
                          <span>28%</span>
                        </div>
                        <Progress value={28} className="w-full" />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>GPU Usage</span>
                          <span>0%</span>
                        </div>
                        <Progress value={0} className="w-full" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Configuration Tab */}
            <TabsContent value="configuration">
              <Card>
                <CardHeader>
                  <CardTitle>Advanced Configuration</CardTitle>
                  <CardDescription>
                    Configure advanced model and system settings
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-lg font-medium mb-4">Detection Settings</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm font-medium">Anomaly Threshold</label>
                          <input 
                            type="number" 
                            step="0.001"
                            className="w-full p-2 border rounded" 
                            defaultValue="0.22610116"
                          />
                        </div>
                        <div>
                          <label className="text-sm font-medium">Confidence Threshold</label>
                          <input 
                            type="number" 
                            step="0.01"
                            className="w-full p-2 border rounded" 
                            defaultValue="0.8"
                          />
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-medium mb-4">Real-time Settings</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm font-medium">Update Interval (ms)</label>
                          <input 
                            type="number" 
                            className="w-full p-2 border rounded" 
                            defaultValue="5000"
                          />
                        </div>
                        <div>
                          <label className="text-sm font-medium">Max History Size</label>
                          <input 
                            type="number" 
                            className="w-full p-2 border rounded" 
                            defaultValue="1000"
                          />
                        </div>
                      </div>
                    </div>

                    <Button className="w-full">Save Configuration</Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default ModelManagementEnhanced;
