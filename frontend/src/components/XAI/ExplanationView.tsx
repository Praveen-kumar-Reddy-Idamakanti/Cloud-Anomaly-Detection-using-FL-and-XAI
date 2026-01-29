
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

interface ExplanationViewProps {
  explanation: ExplanationData | null;
  isLoading?: boolean;
}

const ExplanationView: React.FC<ExplanationViewProps> = ({ 
  explanation, 
  isLoading = false,
  modelSource = 'mock' // Default to mock data for now
}) => {
  // Only process data if explanation exists
  const shapData = explanation?.feature_importances?.map(item => ({
    name: item.feature.replace('_', ' '),
    value: parseFloat((item.importance).toFixed(4)) // Display raw importance, not percentage for SHAP values
  })) || [];

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
                  formatter={(value) => [value, 'Importance']}
                  contentStyle={{ 
                    backgroundColor: 'var(--card)', 
                    borderColor: 'var(--border)',
                    color: 'var(--card-foreground)'
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
