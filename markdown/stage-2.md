# Stage 2 Implementation Plan: Anomaly Classification

## Overview

**Stage 2** adds **attack type classification** to the existing anomaly detection system. While Stage 1 detects "is this an anomaly?", Stage 2 answers "what type of attack is this?".

This document provides a **phased implementation plan** to complete Stage 2 in the frontend.

---

## Current State Analysis

### ✅ Already Implemented (Stage 1)
- Anomaly detection with severity levels
- SHAP feature importance for anomaly detection
- Basic anomaly details (IP, protocol, confidence)
- XAI explanations for anomaly detection

### ❌ Missing (Stage 2)
- Attack type display in UI
- Attack confidence visualization
- Two-stage pipeline visualization
- Attack-type specific XAI explanations
- Integration with enhanced detection endpoint

---

## Phase-Based Implementation Plan

## Phase 1: Backend Integration & Data Models

### 1.1 Update Frontend Data Types
**File**: `frontend/src/api/api.ts`

**Tasks**:
- Extend `AnomalyData` type to properly display attack information
- Add new types for two-stage detection results
- Update API calls to use enhanced endpoint

**Changes Needed**:
```typescript
// Extend AnomalyData type
export type AnomalyData = {
  id: string;
  timestamp: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  sourceIp: string;
  destinationIp: string;
  protocol: string;
  action: string;
  confidence: number;
  reviewed: boolean;
  details: string;
  features?: string;
  // NEW: Add attack classification fields
  attackType?: {
    id: number;
    name: string;
    description: string;
    severity: string;
    color: string;
  };
  attackConfidence?: number;
  reconstructionError?: number;
};

// Add Two-Stage Detection Result type
export type TwoStageDetectionResult = {
  id: string;
  timestamp: Date;
  features: number[];
  isAnomaly: boolean;
  anomalyScore: number;
  threshold: number;
  // Stage 2 results
  attackType?: AttackTypeInfo;
  attackConfidence?: number;
  confidence: number;
};
```

### 1.2 Update API Integration
**Tasks**:
- Modify `modelApi.detectAnomaliesEnhanced()` to return proper types
- Add attack type mapping utilities
- Update detection result formatting

---

## Phase 2: UI Components for Attack Classification

### 2.1 Create Attack Type Badge Component
**File**: `frontend/src/components/AttackTypeBadge.tsx`

**Purpose**: Display attack type with color coding and confidence

**Implementation**:
```typescript
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
    return <Badge variant="outline">Normal</Badge>;
  }

  return (
    <div className="flex items-center space-x-2">
      <Badge 
        variant="outline" 
        style={{ borderColor: attackType.color, color: attackType.color }}
        className={size === 'sm' ? 'text-xs' : size === 'lg' ? 'text-base' : 'text-sm'}
      >
        {attackType.name}
      </Badge>
      {confidence && (
        <span className="text-xs text-muted-foreground">
          {(confidence * 100).toFixed(1)}% confidence
        </span>
      )}
    </div>
  );
};
```

### 2.2 Create Two-Stage Detection Panel
**File**: `frontend/src/components/TwoStagePanel.tsx`

**Purpose**: Show both stages of detection pipeline

**Implementation**:
```typescript
interface TwoStagePanelProps {
  anomalyResult: TwoStageDetectionResult;
}

const TwoStagePanel: React.FC<TwoStagePanelProps> = ({ anomalyResult }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Shield className="mr-2 h-5 w-5 text-blue-500" />
          Two-Stage Detection Pipeline
        </CardTitle>
        <CardDescription>
          Anomaly detection followed by attack type classification
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Stage 1: Anomaly Detection */}
          <div className="border-l-4 border-green-500 pl-4">
            <h4 className="font-medium text-sm mb-2">Stage 1: Anomaly Detection</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Result:</span>
                <span className={`ml-2 ${anomalyResult.isAnomaly ? 'text-red-500' : 'text-green-500'}`}>
                  {anomalyResult.isAnomaly ? 'Anomaly Detected' : 'Normal Traffic'}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Score:</span>
                <span className="ml-2 font-mono">{anomalyResult.anomalyScore.toFixed(4)}</span>
              </div>
            </div>
          </div>

          {/* Stage 2: Attack Classification */}
          <div className={`border-l-4 pl-4 ${anomalyResult.isAnomaly ? 'border-orange-500' : 'border-gray-300'}`}>
            <h4 className="font-medium text-sm mb-2">Stage 2: Attack Classification</h4>
            {anomalyResult.isAnomaly && anomalyResult.attackType ? (
              <div className="space-y-2">
                <AttackTypeBadge 
                  attackType={anomalyResult.attackType} 
                  confidence={anomalyResult.attackConfidence} 
                />
                <p className="text-sm text-muted-foreground">
                  {anomalyResult.attackType.description}
                </p>
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">
                {anomalyResult.isAnomaly ? 'Attack classification processing...' : 'No classification needed'}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
```

---

## Phase 3: Update Existing Pages

### 3.1 Enhance Anomaly Details Page
**File**: `frontend/src/pages/XAIExplanation.tsx`

**Tasks**:
- Add attack type information to anomaly details
- Include Two-Stage Detection Panel
- Update layout to show both stages

**Changes**:
```typescript
// Add to the anomaly details section
{anomaly.attackType && (
  <div className="space-y-1">
    <p className="text-sm text-muted-foreground">Attack Type</p>
    <AttackTypeBadge 
      attackType={anomaly.attackType} 
      confidence={anomaly.attackConfidence} 
    />
  </div>
)}

// Add Two-Stage Panel after anomaly details
<TwoStagePanel anomalyResult={detectionResult} />
```

### 3.2 Update Anomaly List Page
**File**: `frontend/src/pages/XAI.tsx`

**Tasks**:
- Add attack type column to anomaly list
- Show attack confidence in list items
- Update filtering/sorting options

**Changes**:
```typescript
// In the anomaly list item, add attack type info
<div className="flex items-center space-x-2 mb-1">
  <Badge className={getSeverityColor(anomaly.severity)} variant="outline">
    {anomaly.severity.charAt(0).toUpperCase() + anomaly.severity.slice(1)}
  </Badge>
  {anomaly.attackType && (
    <AttackTypeBadge 
      attackType={anomaly.attackType} 
      confidence={anomaly.attackConfidence} 
      size="sm"
    />
  )}
  <span className="text-sm text-muted-foreground">
    {anomaly.sourceIp} → {anomaly.destinationIp}
  </span>
</div>
```

---

## Phase 4: Enhanced XAI Explanations

### 4.1 Create Attack Type Explanation Component
**File**: `frontend/src/components/AttackTypeExplanation.tsx`

**Purpose**: Show attack-type specific XAI explanations

**Implementation**:
```typescript
interface AttackTypeExplanationProps {
  attackType: AttackTypeInfo;
  confidence: number;
  featureImportances: FeatureImportance[];
  contributingFactors: string[];
}

const AttackTypeExplanation: React.FC<AttackTypeExplanationProps> = ({
  attackType,
  confidence,
  featureImportances,
  contributingFactors
}) => {
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
            <div className="w-full bg-secondary rounded-full h-2">
              <div
                className="bg-orange-500 h-2 rounded-full"
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
            <span className="text-sm text-muted-foreground">
              {(confidence * 100).toFixed(1)}% confident this is {attackType.name}
            </span>
          </div>

          {/* Key Indicators for This Attack Type */}
          <div>
            <h4 className="font-medium text-sm mb-2">Key Attack Indicators</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {featureImportances.slice(0, 6).map((feature, index) => (
                <div key={index} className="border rounded p-2">
                  <div className="text-xs font-medium">{feature.feature}</div>
                  <div className="text-xs text-muted-foreground">{feature.description}</div>
                  <div className="text-xs font-mono">
                    Impact: <span className={feature.importance > 0 ? 'text-red-500' : 'text-green-500'}>
                      {feature.importance.toFixed(3)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Contributing Factors */}
          {contributingFactors.length > 0 && (
            <div>
              <h4 className="font-medium text-sm mb-2">Contributing Factors</h4>
              <ul className="space-y-1">
                {contributingFactors.map((factor, index) => (
                  <li key={index} className="flex items-start text-sm">
                    <AlertTriangle className="h-4 w-4 text-orange-500 mr-2 mt-0.5" />
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
```

### 4.2 Update Explanation View Component
**File**: `frontend/src/components/XAI/ExplanationView.tsx`

**Tasks**:
- Add attack type explanation section
- Show two-stage explanation flow
- Integrate with existing SHAP explanations

**Changes**:
```typescript
// Add after existing SHAP explanation
{explanation.attack_type_explanation && (
  <AttackTypeExplanation
    attackType={attackTypeInfo}
    confidence={explanation.attack_confidence}
    featureImportances={explanation.attack_feature_importances || []}
    contributingFactors={explanation.contributing_factors || []}
  />
)}
```

---

## Phase 5: API Integration & Testing

### 5.1 Update API Calls
**Tasks**:
- Use `/model/detect-enhanced` endpoint for new detections
- Update real-time anomaly generation to include attack types
- Add attack type filtering in anomaly list

### 5.2 Add Mock Data for Testing
**File**: `frontend/src/data/mockData.ts`

**Tasks**:
- Add attack type information to mock anomalies
- Create two-stage detection mock results
- Add attack-type specific XAI explanations

### 5.3 Integration Testing
**Tasks**:
- Test end-to-end two-stage detection flow
- Verify attack type display in all components
- Test XAI explanations for attack classification
- Validate responsive design for new components

---

## Phase 6: Documentation & Deployment

### 6.1 Update Documentation
**Tasks**:
- Update frontend README with Stage 2 features
- Add component documentation
- Create usage examples for two-stage API

### 6.2 Performance Optimization
**Tasks**:
- Optimize rendering of attack type components
- Add loading states for attack classification
- Implement error handling for classification failures

---

## Implementation Priority

### High Priority (Core Functionality)
1. **Phase 1**: Data types and API integration
2. **Phase 2**: Attack Type Badge component
3. **Phase 3**: Update anomaly details page
4. **Phase 4**: Basic attack type explanation

### Medium Priority (Enhanced Features)
5. **Phase 4**: Advanced XAI explanations
6. **Phase 5**: Complete API integration
7. **Phase 2**: Two-Stage Detection Panel

### Low Priority (Polish)
8. **Phase 6**: Documentation and optimization
9. **Phase 5**: Advanced filtering and search
10. **Phase 6**: Performance optimizations

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Attack types displayed in anomaly list
- [ ] Attack type details shown in anomaly page
- [ ] Basic attack confidence visualization
- [ ] Integration with enhanced detection endpoint

### Complete Implementation
- [ ] Full two-stage pipeline visualization
- [ ] Attack-type specific XAI explanations
- [ ] Advanced filtering by attack type
- [ ] Comprehensive documentation
- [ ] Performance optimizations
- [ ] Error handling and loading states

---

## Estimated Timeline

- **Phase 1-2**: 2-3 days (Core components)
- **Phase 3-4**: 3-4 days (Page updates and XAI)
- **Phase 5**: 2-3 days (API and testing)
- **Phase 6**: 1-2 days (Documentation and polish)

**Total Estimated Time**: 8-12 days

---

## Dependencies & Prerequisites

### Technical Dependencies
- Existing backend `/model/detect-enhanced` endpoint
- Attack type mapping in backend
- Two-stage XAI explanations in backend

### Skills Required
- React/TypeScript component development
- API integration and error handling
- UI/UX design for complex data visualization
- Testing and debugging

---

## Risk Assessment & Mitigation

### High Risk
- **Backend API Changes**: Enhanced endpoint might not be ready
  - **Mitigation**: Create fallback to standard endpoint
- **Complex Data Flow**: Two-stage detection adds complexity
  - **Mitigation**: Implement incrementally with clear separation

### Medium Risk
- **UI Performance**: Additional components might impact performance
  - **Mitigation**: Lazy loading and optimization
- **Mock Data Alignment**: Mock data might not match real API
  - **Mitigation**: Regular sync with backend team

---

## Next Steps

1. **Start with Phase 1**: Update data types and API integration
2. **Create Core Components**: Attack Type Badge and Two-Stage Panel
3. **Incremental Testing**: Test each phase before proceeding
4. **Regular Reviews**: Weekly progress reviews with stakeholders
5. **Documentation**: Update documentation as features are implemented

---

*This implementation plan provides a structured approach to adding Stage 2 (anomaly classification) to the frontend, ensuring comprehensive coverage of all aspects from basic display to advanced XAI explanations.*
