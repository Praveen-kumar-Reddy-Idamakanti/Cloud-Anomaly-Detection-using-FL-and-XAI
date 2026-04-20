import React from 'react';
import { Badge } from '@/components/ui/badge';
import { AttackTypeInfo } from '../api/api';

interface AttackTypeBadgeProps {
  attackType?: AttackTypeInfo;
  confidence?: number;
  size?: 'sm' | 'md' | 'lg';
}

const AttackTypeBadge: React.FC<AttackTypeBadgeProps> = ({ 
  attackType, 
  confidence, 
  size = 'md' 
}) => {
  if (!attackType) {
    return (
      <Badge 
        variant="outline" 
        className={size === 'sm' ? 'text-xs' : size === 'lg' ? 'text-base' : 'text-sm'}
      >
        Normal
      </Badge>
    );
  }

  const sizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base'
  };

  return (
    <div className="flex items-center space-x-2">
      <Badge 
        variant="outline" 
        style={{ 
          borderColor: attackType.color, 
          color: attackType.color,
          backgroundColor: `${attackType.color}10`
        }}
        className={sizeClasses[size]}
      >
        {attackType.name}
      </Badge>
      {confidence !== undefined && (
        <span className="text-xs text-muted-foreground ml-2">
          {(confidence * 100).toFixed(1)}% confidence
        </span>
      )}
    </div>
  );
};

export default AttackTypeBadge;
