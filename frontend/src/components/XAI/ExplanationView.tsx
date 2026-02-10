
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  Legend,
  LabelList 
} from 'recharts';
import { 
  AlertTriangle, 
  BookOpen, 
  Lightbulb, 
  ShieldCheck, 
  Info, 
  Construction 
} from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  TimeSeriesData, 
  ExplanationData, 
  FeatureImportance 
} from '../../api/api';

// Network traffic feature descriptions for security analysts
const FEATURE_DESCRIPTIONS: { [key: string]: string } = {
  "Flow Duration": "Total duration of the flow in microseconds",
  "Total Fwd Packets": "Total number of forward packets in the flow",
  "Total Backward Packets": "Total number of backward packets in the flow",
  "Total Length of Fwd Packets": "Total size of forward packets in bytes",
  "Total Length of Bwd Packets": "Total size of backward packets in bytes",
  "Flow Bytes/s": "Flow rate in bytes per second",
  "Flow Packets/s": "Flow rate in packets per second",
  "Flow IAT Mean": "Mean time between two consecutive packets",
  "Flow IAT Std": "Standard deviation of time between two packets",
  "Flow IAT Max": "Maximum time between two consecutive packets",
  "Flow IAT Min": "Minimum time between two consecutive packets",
  "Fwd IAT Mean": "Mean time between two consecutive forward packets",
  "Bwd IAT Mean": "Mean time between two consecutive backward packets",
  "Fwd Header Length": "Total size of forward packet headers",
  "Bwd Header Length": "Total size of backward packet headers",
  "Min Packet Length": "Minimum packet size in the flow",
  "Max Packet Length": "Maximum packet size in the flow",
  "Packet Length Mean": "Mean packet size in the flow",
  "Packet Length Std": "Standard deviation of packet sizes",
  "Packet Length Variance": "Variance of packet sizes",
  "FIN Flag Count": "Number of FIN flags in forward packets",
  "SYN Flag Count": "Number of SYN flags in forward packets",
  "RST Flag Count": "Number of RST flags in forward packets",
  "PSH Flag Count": "Number of PSH flags in forward packets",
  "ACK Flag Count": "Number of ACK flags in forward packets",
  "URG Flag Count": "Number of URG flags in forward packets",
  "Down/Up Ratio": "Ratio of download to upload traffic",
  "Average Packet Size": "Average size of packets in the flow",
  "Fwd Segment Size Avg": "Average size of forward segments",
  "Bwd Segment Size Avg": "Average size of backward segments",
  "Active Mean": "Mean active time before becoming idle",
  "Active Std": "Standard deviation of active time",
  "Active Max": "Maximum active time",
  "Active Min": "Minimum active time",
  "Idle Mean": "Mean idle time before becoming active",
  "Idle Std": "Standard deviation of idle time",
  "Idle Max": "Maximum idle time",
  "Idle Min": "Minimum idle time"
};

interface ExplanationViewProps {
  explanation: ExplanationData | null;
  isLoading?: boolean;
}

const ExplanationView: React.FC<ExplanationViewProps> = ({ 
  explanation, 
  isLoading = false
}) => {
  // Only process data if explanation exists
  const shapData = explanation?.feature_importances?.map((item: any, index: number) => {
    // Use the feature name directly from backend (already formatted)
    const featureName = item.feature || `Feature ${index + 1}`;
    
    return {
      name: featureName,
      value: parseFloat((item.importance).toFixed(4)), // Display raw importance, not percentage for SHAP values
      featureIndex: item.feature_index || index, // Include feature index for reference
      description: FEATURE_DESCRIPTIONS[featureName] || "Network traffic feature"
    };
  }) || [];

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="md:col-span-2">
          <CardHeader>
            <div className="h-7 bg-muted/30 animate-pulse rounded-md w-1/3 mb-2" />
            <div className="h-5 bg-muted/30 animate-pulse rounded-md w-1/2" />
          </CardHeader>
          <CardContent>
            <div className="h-64 bg-muted/30 animate-pulse rounded-md" />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <div className="h-7 bg-muted/30 animate-pulse rounded-md w-1/3" />
          </CardHeader>
          <CardContent>
            <div className="h-40 bg-muted/30 animate-pulse rounded-md" />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <div className="h-7 bg-muted/30 animate-pulse rounded-md w-1/3" />
          </CardHeader>
          <CardContent>
            <div className="h-40 bg-muted/30 animate-pulse rounded-md" />
          </CardContent>
        </Card>
      </div>
    );
  }

  // Show a placeholder if explanation is null but not loading
  if (!explanation) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5 text-amber-500" />
              No Explanation Available
            </CardTitle>
            <CardDescription>
              Explanation data for this anomaly could not be loaded.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
              <AlertTriangle className="h-12 w-12 mb-4" />
              <p>Explanation data is missing or could not be processed.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card className="md:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center">
            <BookOpen className="mr-2 h-5 w-5 text-cyberpurple" />
            {explanation.explanation_type} Feature Importance
          </CardTitle>
          <CardDescription>
            Model Type: {explanation.model_type}. {explanation.note}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={shapData}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis 
                  type="number" 
                  tick={{ fill: 'var(--muted-foreground)' }}
                  axisLine={{ stroke: 'var(--border)' }}
                  tickLine={{ stroke: 'var(--border)' }}
                />
                <YAxis 
                  dataKey="name" 
                  type="category" 
                  tick={{ fill: 'var(--muted-foreground)' }}
                  axisLine={{ stroke: 'var(--border)' }}
                  tickLine={{ stroke: 'var(--border)' }}
                />
                <Tooltip 
                  formatter={(value, name, props: any) => [
                    value, 
                    'SHAP Value'
                  ]}
                  labelFormatter={(label) => {
                    const featureData = shapData.find(item => item.name === label);
                    return `${label}${featureData?.description ? ': ' + featureData.description : ''}`;
                  }}
                  contentStyle={{ 
                    backgroundColor: 'var(--card)', 
                    borderColor: 'var(--border)',
                    color: 'var(--card-foreground)',
                    maxWidth: '300px'
                  }}
                  labelStyle={{ color: 'var(--card-foreground)' }}
                />
                <Bar dataKey="value" fill="#8B5CF6">
                  <LabelList dataKey="value" position="right" formatter={(value: number) => value.toFixed(4)} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
      
      {/* Feature Description Card */}
      {shapData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Info className="mr-2 h-5 w-5 text-blue-500" />
              Top Feature Descriptions
            </CardTitle>
            <CardDescription>
              Understanding the most important network traffic features
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {shapData.slice(0, 5).map((feature, index) => (
                <div key={feature.featureIndex} className="border-l-4 border-purple-500 pl-3">
                  <div className="font-medium text-sm">{feature.name}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {feature.description}
                  </div>
                  <div className="text-xs font-mono mt-1">
                    SHAP Value: <span className={feature.value > 0 ? 'text-red-500' : 'text-green-500'}>
                      {feature.value.toFixed(4)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
      
      {explanation.contributingFactors && explanation.contributingFactors.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5 text-amber-500" />
              Contributing Factors
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {explanation.contributingFactors.map((factor, index) => (
                <li key={index} className="flex items-start">
                  <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-amber-500/10 text-amber-500 mr-2">
                    {index + 1}
                  </span>
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
      
      {explanation.recommendations && explanation.recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <ShieldCheck className="mr-2 h-5 w-5 text-green-500" />
              Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {explanation.recommendations.map((recommendation, index) => (
                <li key={index} className="flex items-start">
                  <Lightbulb className="h-5 w-5 text-green-500 mr-2 shrink-0" />
                  <span>{recommendation}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ExplanationView;
