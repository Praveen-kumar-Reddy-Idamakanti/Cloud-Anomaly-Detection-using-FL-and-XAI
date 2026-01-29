import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { Brain, Target, Zap, TrendingUp, RefreshCw } from 'lucide-react';
import { modelApi } from '../../api/api';

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  truePositiveRate: number;
  trueNegativeRate: number;
}

interface AttackTypeMetrics {
  attackType: string;
  count: number;
  accuracy: number;
  avgConfidence: number;
}

interface TrainingHistory {
  round: number;
  loss: number;
  accuracy: number;
  timestamp: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

export const ModelPerformanceMetrics: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [attackTypeMetrics, setAttackTypeMetrics] = useState<AttackTypeMetrics[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistory[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');

  useEffect(() => {
    fetchModelData();
    const interval = setInterval(fetchModelData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchModelData = async () => {
    try {
      const info = await modelApi.getModelInfo();
      setModelInfo(info);
      
      // Generate mock metrics based on model info
      const mockMetrics: ModelMetrics = {
        accuracy: info.accuracy || 0.87,
        precision: 0.89,
        recall: 0.85,
        f1Score: 0.87,
        falsePositiveRate: 0.08,
        falseNegativeRate: 0.15,
        truePositiveRate: 0.85,
        trueNegativeRate: 0.92
      };
      setMetrics(mockMetrics);

      // Generate mock attack type metrics
      const mockAttackMetrics: AttackTypeMetrics[] = [
        { attackType: 'BENIGN', count: 850, accuracy: 0.95, avgConfidence: 0.92 },
        { attackType: 'DoS GoldenEye', count: 45, accuracy: 0.88, avgConfidence: 0.85 },
        { attackType: 'DoS Hulk', count: 38, accuracy: 0.91, avgConfidence: 0.87 },
        { attackType: 'DoS Slowhttptest', count: 28, accuracy: 0.84, avgConfidence: 0.82 },
        { attackType: 'DoS slowloris', count: 22, accuracy: 0.86, avgConfidence: 0.83 }
      ];
      setAttackTypeMetrics(mockAttackMetrics);

      // Generate mock training history
      const mockHistory: TrainingHistory[] = Array.from({ length: 20 }, (_, i) => ({
        round: i + 1,
        loss: 0.8 - (i * 0.03) + Math.random() * 0.1,
        accuracy: 0.7 + (i * 0.01) + Math.random() * 0.05,
        timestamp: new Date(Date.now() - (19 - i) * 3600000).toISOString()
      }));
      setTrainingHistory(mockHistory);

    } catch (error) {
      console.error('Failed to fetch model data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;

  const getMetricColor = (value: number, type: 'accuracy' | 'error' = 'accuracy') => {
    if (type === 'accuracy') {
      if (value >= 0.9) return 'text-green-600';
      if (value >= 0.8) return 'text-yellow-600';
      return 'text-red-600';
    } else {
      if (value <= 0.1) return 'text-green-600';
      if (value <= 0.2) return 'text-yellow-600';
      return 'text-red-600';
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Brain className="h-5 w-5 mr-2" />
            Model Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <RefreshCw className="h-8 w-8 animate-spin" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Accuracy</p>
                <p className={`text-2xl font-bold ${getMetricColor(metrics?.accuracy || 0)}`}>
                  {formatPercentage(metrics?.accuracy || 0)}
                </p>
              </div>
              <Target className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">F1 Score</p>
                <p className={`text-2xl font-bold ${getMetricColor(metrics?.f1Score || 0)}`}>
                  {formatPercentage(metrics?.f1Score || 0)}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">False Positive Rate</p>
                <p className={`text-2xl font-bold ${getMetricColor(metrics?.falsePositiveRate || 0, 'error')}`}>
                  {formatPercentage(metrics?.falsePositiveRate || 0)}
                </p>
              </div>
              <Zap className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Model Status</p>
                <p className="text-lg font-bold">
                  <Badge variant={modelInfo?.status === 'loaded' ? 'default' : 'secondary'}>
                    {modelInfo?.status || 'Unknown'}
                  </Badge>
                </p>
              </div>
              <Brain className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Metrics Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { label: 'Precision', value: metrics?.precision || 0 },
                { label: 'Recall', value: metrics?.recall || 0 },
                { label: 'True Positive Rate', value: metrics?.truePositiveRate || 0 },
                { label: 'True Negative Rate', value: metrics?.trueNegativeRate || 0 },
                { label: 'False Positive Rate', value: metrics?.falsePositiveRate || 0 },
                { label: 'False Negative Rate', value: metrics?.falseNegativeRate || 0 }
              ].map((metric) => (
                <div key={metric.label} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{metric.label}</span>
                    <span className={getMetricColor(metric.value, metric.label.includes('False') ? 'error' : 'accuracy')}>
                      {formatPercentage(metric.value)}
                    </span>
                  </div>
                  <Progress value={metric.value * 100} className="h-2" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Attack Type Performance */}
        <Card>
          <CardHeader>
            <CardTitle>Attack Type Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={attackTypeMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="attackType" angle={-45} textAnchor="end" height={80} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="accuracy" fill="#8884d8" name="Accuracy" />
                <Bar dataKey="avgConfidence" fill="#82ca9d" name="Avg Confidence" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Training History */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Training History
            <div className="flex items-center space-x-2">
              <select 
                value={selectedTimeRange} 
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="text-sm border rounded px-2 py-1"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <Button variant="outline" size="sm" onClick={fetchModelData}>
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="round" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Bar yAxisId="left" dataKey="loss" fill="#ff7300" name="Loss" />
              <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#8884d8" name="Accuracy" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Attack Type Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Attack Type Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={attackTypeMetrics}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ attackType, count }) => `${attackType}: ${count}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {attackTypeMetrics.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Model Type:</span>
                <span className="text-sm">
                  {modelInfo?.two_stage_enabled ? 'Two-Stage Enhanced' : 'Standard'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Input Dimensions:</span>
                <span className="text-sm">{modelInfo?.input_dim || 0} features</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Attack Types:</span>
                <span className="text-sm">{modelInfo?.attack_types?.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Last Trained:</span>
                <span className="text-sm">
                  {modelInfo?.last_trained ? new Date(modelInfo.last_trained).toLocaleDateString() : 'Unknown'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Model Path:</span>
                <span className="text-sm font-mono text-xs">
                  {modelInfo?.model_path?.split('/').pop() || 'Unknown'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
