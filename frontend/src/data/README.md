# Network Activity Mock Data & Visualization Utils

This directory contains mock data and utilities for network activity visualization in the dashboard.

## Files

### `networkActivityMockData.ts`
Contains various mock data collections for network activity visualization:

#### Data Collections:
- **`networkActivityMockData`**: 30 days of network activity with realistic patterns
- **`realtimeNetworkActivity`**: Last 24 hours of network activity (hourly data)
- **`weeklyComparisonData`**: Weekly aggregated data for comparison
- **`hourlyPatternData`**: Typical hourly network activity pattern
- **`dailyPatternData`**: Network activity by day of week

#### Data Structure:
```typescript
interface TimeSeriesData {
  date: string;        // Date in YYYY-MM-DD format
  logs: number;        // Network activity count
  anomalies: number;   // Anomaly detection count
}
```

#### Dynamic Data Generation:
- **`generateNetworkActivityData(days, baseLogs, baseAnomalies)`**: Generate custom network activity data

### `networkVisualizationUtils.ts`
Utilities for network activity visualization and analysis:

#### Scenarios:
- **NORMAL**: Normal daily variation
- **SPIKE**: Normal with occasional traffic spikes
- **GRADUAL_INCREASE**: Gradual increase over time
- **CYCLIC**: Weekly patterns (weekdays vs weekends)
- **ANOMALY_BURST**: Normal with anomaly bursts

#### Functions:
- **`generateScenarioData(scenario, days)`**: Generate data based on scenario
- **`calculateNetworkStats(data)`**: Calculate network statistics
- **`detectAnomalousPatterns(data)`**: Detect anomalous patterns
- **`formatForChart`**: Format data for different chart types

## Usage Examples

### Basic Usage:
```typescript
import { networkActivityMockData } from './networkActivityMockData';

// Use in dashboard
const data = networkActivityMockData;
```

### Scenario-based Data:
```typescript
import { generateScenarioData, NETWORK_SCENARIOS } from './networkVisualizationUtils';

// Generate spike scenario data
const spikeData = generateScenarioData('SPIKE', 30);
```

### Statistics Calculation:
```typescript
import { calculateNetworkStats } from './networkVisualizationUtils';

const stats = calculateNetworkStats(data);
console.log(stats.avgLogs, stats.anomalyRate);
```

### Pattern Detection:
```typescript
import { detectAnomalousPatterns } from './networkVisualizationUtils';

const patterns = detectAnomalousPatterns(data);
patterns.forEach(pattern => {
  console.log(`${pattern.date}: ${pattern.description}`);
});
```

## Integration with Dashboard

The mock data is automatically used as a fallback when the API fails. The API integration in `api.ts`:

```typescript
getTimeSeriesData: async (): Promise<TimeSeriesData[]> => {
  try {
    // Try real API first
    const data = await apiCall('/history/training');
    return transformData(data);
  } catch (error) {
    // Fallback to mock data
    const { networkActivityMockData } = await import('../data/networkActivityMockData');
    return networkActivityMockData;
  }
}
```

## Chart Customization

### Colors:
- Network Activity: Blue (#8884d8)
- Anomalies Detected: Red (#ff4557)

### Chart Types:
- **Area Chart**: Best for continuous data visualization
- **Bar Chart**: Good for comparative analysis
- **Line Chart**: Ideal for trend analysis

### Data Formatting:
```typescript
import { formatForChart } from './networkVisualizationUtils';

// Format for different chart types
const areaData = formatForChart.area(data);
const barData = formatForChart.bar(data);
const lineData = formatForChart.line(data);
```

## Data Patterns

### Realistic Network Activity:
- **Daily cycles**: Higher activity during business hours
- **Weekly patterns**: Weekdays typically show higher activity
- **Seasonal variations**: Long-term trends and patterns
- **Anomaly correlation**: Anomaly counts often correlate with activity levels

### Anomaly Patterns:
- **Volume spikes**: Sudden increases in network traffic
- **Anomaly bursts**: Periods of high anomaly detection
- **Unusual patterns**: Deviations from normal behavior

## Performance Considerations

- Mock data is static and loads instantly
- Real API calls may have latency
- Consider implementing caching for frequently accessed data
- Use pagination for large datasets

## Future Enhancements

1. **Real-time simulation**: Add real-time data generation
2. **More scenarios**: Additional network activity scenarios
3. **Advanced analytics**: Machine learning-based pattern detection
4. **Export functionality**: Data export capabilities
5. **Custom thresholds**: Configurable anomaly detection thresholds
