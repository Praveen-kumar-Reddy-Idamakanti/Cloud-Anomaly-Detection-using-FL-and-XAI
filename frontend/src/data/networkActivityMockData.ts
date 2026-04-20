import { TimeSeriesData } from '../api/api';

// Mock data for network activity visualization
export const networkActivityMockData: TimeSeriesData[] = [
  {
    date: '2024-01-01',
    logs: 12500,
    anomalies: 45
  },
  {
    date: '2024-01-02',
    logs: 18900,
    anomalies: 78
  },
  {
    date: '2024-01-03',
    logs: 15600,
    anomalies: 62
  },
  {
    date: '2024-01-04',
    logs: 28900,
    anomalies: 128
  },
  {
    date: '2024-01-05',
    logs: 22100,
    anomalies: 95
  },
  {
    date: '2024-01-06',
    logs: 35700,
    anomalies: 178
  },
  {
    date: '2024-01-07',
    logs: 41200,
    anomalies: 212
  },
  {
    date: '2024-01-08',
    logs: 27600,
    anomalies: 118
  },
  {
    date: '2024-01-09',
    logs: 32400,
    anomalies: 156
  },
  {
    date: '2024-01-10',
    logs: 49200,
    anomalies: 275
  },
  {
    date: '2024-01-11',
    logs: 38900,
    anomalies: 195
  },
  {
    date: '2024-01-12',
    logs: 52100,
    anomalies: 298
  },
  {
    date: '2024-01-13',
    logs: 45800,
    anomalies: 248
  },
  {
    date: '2024-01-14',
    logs: 60100,
    anomalies: 368
  },
  {
    date: '2024-01-15',
    logs: 31200,
    anomalies: 145
  },
  {
    date: '2024-01-16',
    logs: 56700,
    anomalies: 335
  },
  {
    date: '2024-01-17',
    logs: 63400,
    anomalies: 395
  },
  {
    date: '2024-01-18',
    logs: 46700,
    anomalies: 258
  },
  {
    date: '2024-01-19',
    logs: 72300,
    anomalies: 478
  },
  {
    date: '2024-01-20',
    logs: 53400,
    anomalies: 312
  },
  {
    date: '2024-01-21',
    logs: 58900,
    anomalies: 356
  },
  {
    date: '2024-01-22',
    logs: 65600,
    anomalies: 418
  },
  {
    date: '2024-01-23',
    logs: 48900,
    anomalies: 276
  },
  {
    date: '2024-01-24',
    logs: 68900,
    anomalies: 445
  },
  {
    date: '2024-01-25',
    logs: 61200,
    anomalies: 378
  },
  {
    date: '2024-01-26',
    logs: 39800,
    anomalies: 198
  },
  {
    date: '2024-01-27',
    logs: 57600,
    anomalies: 342
  },
  {
    date: '2024-01-28',
    logs: 69800,
    anomalies: 462
  },
  {
    date: '2024-01-29',
    logs: 74500,
    anomalies: 512
  },
  {
    date: '2024-01-30',
    logs: 81200,
    anomalies: 568
  }
];

// Real-time network activity data (last 24 hours)
export const realtimeNetworkActivity: TimeSeriesData[] = [
  {
    date: '2024-01-30 00:00',
    logs: 2890,
    anomalies: 23
  },
  {
    date: '2024-01-30 04:00',
    logs: 2156,
    anomalies: 18
  },
  {
    date: '2024-01-30 08:00',
    logs: 3456,
    anomalies: 31
  },
  {
    date: '2024-01-30 12:00',
    logs: 4567,
    anomalies: 42
  },
  {
    date: '2024-01-30 16:00',
    logs: 5234,
    anomalies: 48
  },
  {
    date: '2024-01-30 20:00',
    logs: 4123,
    anomalies: 35
  },
  {
    date: '2024-01-30 23:59',
    logs: 3789,
    anomalies: 29
  }
];

// Weekly comparison data
export const weeklyComparisonData: TimeSeriesData[] = [
  {
    date: 'Week 1',
    logs: 185600,
    anomalies: 892
  },
  {
    date: 'Week 2',
    logs: 234500,
    anomalies: 1234
  },
  {
    date: 'Week 3',
    logs: 298700,
    anomalies: 1567
  },
  {
    date: 'Week 4',
    logs: 356800,
    anomalies: 1987
  }
];

// Hourly network activity pattern (typical day)
export const hourlyPatternData: TimeSeriesData[] = [
  { date: '00:00', logs: 1234, anomalies: 12 },
  { date: '01:00', logs: 987, anomalies: 8 },
  { date: '02:00', logs: 876, anomalies: 6 },
  { date: '03:00', logs: 765, anomalies: 5 },
  { date: '04:00', logs: 654, anomalies: 4 },
  { date: '05:00', logs: 876, anomalies: 7 },
  { date: '06:00', logs: 1234, anomalies: 11 },
  { date: '07:00', logs: 2345, anomalies: 19 },
  { date: '08:00', logs: 3456, anomalies: 28 },
  { date: '09:00', logs: 4567, anomalies: 37 },
  { date: '10:00', logs: 5234, anomalies: 43 },
  { date: '11:00', logs: 5678, anomalies: 47 },
  { date: '12:00', logs: 5890, anomalies: 49 },
  { date: '13:00', logs: 5456, anomalies: 45 },
  { date: '14:00', logs: 5123, anomalies: 42 },
  { date: '15:00', logs: 4789, anomalies: 39 },
  { date: '16:00', logs: 5234, anomalies: 44 },
  { date: '17:00', logs: 5678, anomalies: 48 },
  { date: '18:00', logs: 4567, anomalies: 38 },
  { date: '19:00', logs: 3456, anomalies: 29 },
  { date: '20:00', logs: 2345, anomalies: 20 },
  { date: '21:00', logs: 1876, anomalies: 16 },
  { date: '22:00', logs: 1543, anomalies: 13 },
  { date: '23:00', logs: 1324, anomalies: 11 }
];

// Network activity by day of week
export const dailyPatternData: TimeSeriesData[] = [
  { date: 'Monday', logs: 45678, anomalies: 234 },
  { date: 'Tuesday', logs: 52345, anomalies: 267 },
  { date: 'Wednesday', logs: 49876, anomalies: 256 },
  { date: 'Thursday', logs: 56789, anomalies: 289 },
  { date: 'Friday', logs: 61234, anomalies: 312 },
  { date: 'Saturday', logs: 34567, anomalies: 178 },
  { date: 'Sunday', logs: 28901, anomalies: 145 }
];

// Mock data generator for dynamic scenarios
export const generateNetworkActivityData = (days: number, baseLogs: number = 30000, baseAnomalies: number = 150): TimeSeriesData[] => {
  const data: TimeSeriesData[] = [];
  const today = new Date();
  
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    
    // Add some realistic variation
    const variation = Math.sin(i * 0.5) * 0.3 + Math.random() * 0.2;
    const logs = Math.floor(baseLogs * (1 + variation));
    const anomalies = Math.floor(baseAnomalies * (1 + variation * 0.8));
    
    data.push({
      date: date.toISOString().split('T')[0],
      logs: Math.max(1000, logs),
      anomalies: Math.max(10, anomalies)
    });
  }
  
  return data;
};

// Export all mock data collections
export default {
  networkActivityMockData,
  realtimeNetworkActivity,
  weeklyComparisonData,
  hourlyPatternData,
  dailyPatternData,
  generateNetworkActivityData
};
