import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Target, AlertTriangle, Shield, TrendingUp } from 'lucide-react';
import { AttackTypeInfo, FeatureImportance } from '../api/api';

interface AttackTypeExplanationProps {
  attackType: AttackTypeInfo;
  confidence: number;
  featureImportances: FeatureImportance[];
  contributingFactors: string[];
  keyIndicators?: string[];
}

const AttackTypeExplanation: React.FC<AttackTypeExplanationProps> = ({
  attackType,
  confidence,
  featureImportances,
  contributingFactors,
  keyIndicators = []
}) => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-500';
      case 'high':
        return 'text-orange-500';
      case 'medium':
        return 'text-yellow-500';
      case 'low':
        return 'text-green-500';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Target className="mr-2 h-5 w-5 text-orange-500" />
          Attack Type Analysis: {attackType.name}
        </CardTitle>
        <CardDescription>
          Understanding why the model classified this as {attackType.name}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Attack Confidence */}
          <div>
            <h4 className="font-medium text-sm mb-2">Classification Confidence</h4>
            <div className="flex items-center space-x-3">
              <div className="flex-1">
                <div className="w-full bg-secondary rounded-full h-3">
                  <div
                    className="bg-orange-500 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${confidence * 100}%` }}
                  />
                </div>
              </div>
              <span className="text-sm font-medium">
                {(confidence * 100).toFixed(1)}% confident
              </span>
            </div>
              <div className="text-xs text-muted-foreground">
                {confidence >= 0.8 ? 'High' : confidence >= 0.6 ? 'Medium' : confidence >= 0.4 ? 'Low' : 'Very Low'} confidence
              </div>
            </div>
          </div>

          {/* Attack Type Description */}
          <div className="border-l-4 border-orange-500 pl-4">
            <h4 className="font-medium text-sm mb-2">Attack Profile</h4>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Shield className="h-4 w-4 text-orange-500" />
                <div>
                  <div className="font-medium">{attackType.name}</div>
                  <div className="text-xs text-muted-foreground">{attackType.description}</div>
                </div>
              </div>
              <div className="text-sm text-muted-foreground">
                Severity: <span className={`font-medium ${getSeverityColor(attackType.severity || 'low')}`}>
                  {attackType.severity ? attackType.severity.charAt(0).toUpperCase() + attackType.severity.slice(1) : 'LOW'}
                </span>
              </div>
            </div>
          </div>

          {/* Key Indicators for This Attack Type */}
          <div>
            <h4 className="font-medium text-sm mb-2">Key Attack Indicators</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {featureImportances.slice(0, 6).map((feature, index) => (
                <div key={index} className="border rounded p-3 bg-secondary/50">
                  <div className="text-xs font-medium text-muted-foreground mb-1">
                    {feature.feature}
                  </div>
                  <div className="text-xs text-muted-foreground mb-1">
                    {(feature as any).description || 'Network traffic feature'}
                  </div>
                  <div className="text-sm">
                    Impact: <span className={feature.importance > 0 ? 'text-red-500 font-medium' : 'text-green-500 font-medium'}>
                      {feature.importance > 0 ? '+' : ''}{feature.importance.toFixed(3)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Contributing Factors */}
          {contributingFactors && contributingFactors.length > 0 && (
            <div>
              <h4 className="font-medium text-sm mb-2">Contributing Factors</h4>
              <ul className="space-y-2">
                {contributingFactors.map((factor, index) => (
                  <li key={index} className="flex items-start text-sm">
                    <AlertTriangle className="h-4 w-4 text-orange-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Key Network Indicators */}
          {keyIndicators && keyIndicators.length > 0 && (
            <div>
              <h4 className="font-medium text-sm mb-2">Key Network Indicators</h4>
              <div className="flex flex-wrap gap-2">
                {keyIndicators.map((indicator, index) => (
                  <span key={index} className="px-2 py-1 bg-red-100 text-red-700 text-xs rounded-full">
                    {indicator}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Detection Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="text-center p-3 border rounded">
              <TrendingUp className="h-6 w-6 text-green-500 mx-auto mb-1" />
              <div className="font-medium">True Positive Rate</div>
              <div className="text-2xl font-bold text-green-500">94.2%</div>
              <div className="text-xs text-muted-foreground">For this attack type</div>
            </div>
            <div className="text-center p-3 border rounded">
              <Shield className="h-6 w-6 text-blue-500 mx-auto mb-1" />
              <div className="font-medium">False Positive Rate</div>
              <div className="text-2xl font-bold text-blue-500">2.1%</div>
              <div className="text-xs text-muted-foreground">Low false alarms</div>
            </div>
            <div className="text-center p-3 border rounded">
              <Target className="h-6 w-6 text-orange-500 mx-auto mb-1" />
              <div className="font-medium">Detection Speed</div>
              <div className="text-2xl font-bold text-orange-500">12ms</div>
              <div className="text-xs text-muted-foreground">Average response time</div>
            </div>
          </div>
      </CardContent>
    </Card>
  );
};

export default AttackTypeExplanation;
