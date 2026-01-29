import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { DetectionResult, AttackTypeInfo } from '../../api/api';

interface AttackTypeVisualizationProps {
  results: DetectionResult[];
}

export const AttackTypeVisualization: React.FC<AttackTypeVisualizationProps> = ({
  results
}) => {
  const getAttackTypeStats = () => {
    const stats: Record<string, { count: number; severity: string; color: string }> = {};
    
    results.forEach(result => {
      if (result.isAnomaly && result.attackType) {
        const key = result.attackType.name;
        if (!stats[key]) {
          stats[key] = {
            count: 0,
            severity: result.attackType.severity,
            color: result.attackType.color
          };
        }
        stats[key].count++;
      }
    });
    
    return stats;
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const attackStats = getAttackTypeStats();
  const totalAnomalies = results.filter(r => r.isAnomaly).length;
  const attackTypes = Object.entries(attackStats);

  if (attackTypes.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Attack Type Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            No attack types detected. All traffic appears to be normal.
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Attack Type Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Attack Type Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {attackTypes.map(([attackType, stats]) => {
              const percentage = totalAnomalies > 0 ? (stats.count / totalAnomalies) * 100 : 0;
              
              return (
                <div key={attackType} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full ${getSeverityColor(stats.severity)}`}></div>
                      <span className="font-medium">{attackType}</span>
                      <Badge variant="outline">{stats.count}</Badge>
                    </div>
                    <span className="text-sm text-gray-600">{percentage.toFixed(1)}%</span>
                  </div>
                  <Progress value={percentage} className="h-2" />
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Attack Severity Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle>Severity Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {['critical', 'high', 'medium', 'low'].map(severity => {
              const count = attackTypes
                .filter(([_, stats]) => stats.severity === severity)
                .reduce((sum, [_, stats]) => sum + stats.count, 0);
              
              const percentage = totalAnomalies > 0 ? (count / totalAnomalies) * 100 : 0;
              
              return (
                <div key={severity} className="text-center">
                  <div className={`w-16 h-16 rounded-full ${getSeverityColor(severity)} mx-auto mb-2 flex items-center justify-center`}>
                    <span className="text-white font-bold text-lg">{count}</span>
                  </div>
                  <div className="font-medium capitalize">{severity}</div>
                  <div className="text-sm text-gray-600">{percentage.toFixed(1)}%</div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Recent Attacks Timeline */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Attack Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {results
              .filter(r => r.isAnomaly && r.attackType)
              .slice(-10)
              .reverse()
              .map((result) => (
                <div key={result.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${getSeverityColor(result.attackType?.severity || 'low')}`}></div>
                    <div>
                      <div className="font-medium">{result.attackType?.name}</div>
                      <div className="text-sm text-gray-600">
                        Score: {result.anomalyScore.toFixed(4)}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">
                      {result.attackConfidence && `${(result.attackConfidence * 100).toFixed(1)}% confidence`}
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(result.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>

      {/* Attack Confidence Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Attack Confidence Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {attackTypes.map(([attackType, stats]) => {
              const attackResults = results.filter(
                r => r.isAnomaly && r.attackType?.name === attackType
              );
              
              if (attackResults.length === 0) return null;
              
              const avgConfidence = attackResults.reduce(
                (sum, r) => sum + (r.attackConfidence || 0), 0
              ) / attackResults.length;
              
              const maxConfidence = Math.max(...attackResults.map(r => r.attackConfidence || 0));
              const minConfidence = Math.min(...attackResults.map(r => r.attackConfidence || 0));
              
              return (
                <div key={attackType} className="p-4 border rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">{attackType}</span>
                    <Badge>{stats.count} occurrences</Badge>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <div className="text-gray-600">Average Confidence</div>
                      <div className="font-medium">{(avgConfidence * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-gray-600">Max Confidence</div>
                      <div className="font-medium">{(maxConfidence * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-gray-600">Min Confidence</div>
                      <div className="font-medium">{(minConfidence * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
