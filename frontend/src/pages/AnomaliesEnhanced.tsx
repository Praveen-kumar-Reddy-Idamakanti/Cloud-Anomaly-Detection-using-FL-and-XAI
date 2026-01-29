import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, ChevronLeft, ChevronRight, Filter, RefreshCw, Download, CheckSquare, Square, X } from 'lucide-react';
import { toast } from 'sonner';
import Navbar from '../components/Layout/Navbar';
import Sidebar from '../components/Layout/Sidebar';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { anomaliesApi, getAttackTypeInfo, AttackTypeInfo, AnomalyData } from '../api/api';

interface EnhancedAnomalyData extends AnomalyData {
  attackType?: AttackTypeInfo;
  attackConfidence?: number;
  anomalyScore?: number;
  isAnomaly?: boolean;
}

const AnomaliesEnhanced: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [anomalies, setAnomalies] = useState<EnhancedAnomalyData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalAnomalies, setTotalAnomalies] = useState(0);
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [attackTypeFilter, setAttackTypeFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedAnomalies, setSelectedAnomalies] = useState<Set<string>>(new Set());
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const limit = 10;

  useEffect(() => {
    fetchAnomalies();
  }, [page, severityFilter, attackTypeFilter, statusFilter]);

  const fetchAnomalies = async () => {
    setIsLoading(true);
    try {
      const response = await anomaliesApi.getAnomalies(page, limit);
      
      // Apply filters and add mock attack type data
      let filteredData = response.data.map(anomaly => ({
        ...anomaly,
        // Add mock attack type data for demonstration
        isAnomaly: anomaly.severity !== 'low',
        attackType: anomaly.severity !== 'low' ? getAttackTypeInfo(Math.floor(Math.random() * 4) + 1) : undefined,
        attackConfidence: anomaly.severity !== 'low' ? Math.random() * 0.3 + 0.7 : undefined,
        anomalyScore: anomaly.confidence
      }));
      
      if (severityFilter !== 'all') {
        filteredData = filteredData.filter(
          (anomaly) => anomaly.severity === severityFilter
        );
      }
      
      if (attackTypeFilter !== 'all') {
        filteredData = filteredData.filter(
          (anomaly) => anomaly.attackType?.name === attackTypeFilter
        );
      }
      
      if (statusFilter !== 'all') {
        filteredData = filteredData.filter(
          (anomaly) => (statusFilter === 'reviewed' ? anomaly.reviewed : !anomaly.reviewed)
        );
      }
      
      setAnomalies(filteredData);
      setTotalAnomalies(response.total);
    } catch (error: any) {
      toast.error(error.message || 'Failed to fetch anomalies');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchAnomalies();
  };

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const getSeverityColor = (severity: EnhancedAnomalyData['severity']) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'high':
        return 'bg-amber-500/10 text-amber-500 border-amber-500/20';
      case 'medium':
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      case 'low':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      default:
        return 'bg-secondary text-secondary-foreground';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const totalPages = Math.ceil(totalAnomalies / limit);

  // Selection handling
  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedAnomalies(new Set(anomalies.map(a => a.id)));
    } else {
      setSelectedAnomalies(new Set());
    }
  };

  const handleSelectAnomaly = (id: string, checked: boolean) => {
    const newSelected = new Set(selectedAnomalies);
    if (checked) {
      newSelected.add(id);
    } else {
      newSelected.delete(id);
    }
    setSelectedAnomalies(newSelected);
  };

  // Batch operations
  const handleBatchReview = async (reviewed: boolean) => {
    if (selectedAnomalies.size === 0) {
      toast.error('No anomalies selected');
      return;
    }

    setIsBatchProcessing(true);
    try {
      // Simulate batch review API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast.success(`Marked ${selectedAnomalies.size} anomalies as ${reviewed ? 'reviewed' : 'pending'}`);
      setSelectedAnomalies(new Set());
      fetchAnomalies();
    } catch (error: any) {
      toast.error(error.message || 'Batch review failed');
    } finally {
      setIsBatchProcessing(false);
    }
  };

  // Export functionality
  const handleExport = (format: 'csv' | 'json') => {
    const dataToExport = anomalies.filter(a => selectedAnomalies.has(a.id) || selectedAnomalies.size === 0);
    
    if (dataToExport.length === 0) {
      toast.error('No data to export');
      return;
    }

    let content: string;
    let filename: string;
    let mimeType: string;

    if (format === 'csv') {
      const headers = ['ID', 'Timestamp', 'Severity', 'Source IP', 'Destination IP', 'Protocol', 'Action', 'Confidence', 'Attack Type', 'Status'];
      const rows = dataToExport.map(anomaly => [
        anomaly.id,
        anomaly.timestamp,
        anomaly.severity,
        anomaly.sourceIp,
        anomaly.destinationIp,
        anomaly.protocol,
        anomaly.action,
        anomaly.confidence.toString(),
        anomaly.attackType?.name || 'N/A',
        anomaly.reviewed ? 'Reviewed' : 'Pending'
      ]);
      
      content = [headers, ...rows].map(row => row.join(',')).join('\\n');
      filename = `anomalies_${new Date().toISOString().split('T')[0]}.csv`;
      mimeType = 'text/csv';
    } else {
      content = JSON.stringify(dataToExport, null, 2);
      filename = `anomalies_${new Date().toISOString().split('T')[0]}.json`;
      mimeType = 'application/json';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast.success(`Exported ${dataToExport.length} anomalies as ${format.toUpperCase()}`);
  };

  const uniqueAttackTypes = Array.from(new Set(anomalies.map(a => a.attackType?.name).filter(Boolean)));

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <AlertTriangle className="h-6 w-6 mr-2 text-cyberpurple" />
              <h1 className="text-2xl font-bold">Enhanced Anomaly Management</h1>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" onClick={() => handleExport('csv')}>
                <Download className="h-4 w-4 mr-1" />
                Export CSV
              </Button>
              <Button variant="outline" size="sm" onClick={() => handleExport('json')}>
                <Download className="h-4 w-4 mr-1" />
                Export JSON
              </Button>
            </div>
          </div>

          {/* Statistics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-blue-600">{totalAnomalies}</div>
                <div className="text-sm text-gray-600">Total Anomalies</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-red-600">
                  {anomalies.filter(a => a.severity === 'critical' || a.severity === 'high').length}
                </div>
                <div className="text-sm text-gray-600">High Priority</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-yellow-600">
                  {anomalies.filter(a => !a.reviewed).length}
                </div>
                <div className="text-sm text-gray-600">Pending Review</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-green-600">
                  {uniqueAttackTypes.length}
                </div>
                <div className="text-sm text-gray-600">Attack Types</div>
              </CardContent>
            </Card>
          </div>

          {/* Filters and Batch Operations */}
          <Card className="mb-6">
            <CardContent className="pt-6">
              <div className="flex flex-wrap items-center gap-4">
                {/* Filters */}
                <div className="flex items-center space-x-2">
                  <Filter className="h-4 w-4 text-muted-foreground" />
                  <Select value={severityFilter} onValueChange={setSeverityFilter}>
                    <SelectTrigger className="w-[130px]">
                      <SelectValue placeholder="Severity" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Severities</SelectItem>
                      <SelectItem value="critical">Critical</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Select value={attackTypeFilter} onValueChange={setAttackTypeFilter}>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Attack Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Attack Types</SelectItem>
                    {uniqueAttackTypes.map(type => (
                      <SelectItem key={type} value={type}>{type}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-[120px]">
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="reviewed">Reviewed</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                  </SelectContent>
                </Select>

                <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isLoading}>
                  <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                </Button>

                {/* Batch Operations */}
                {selectedAnomalies.size > 0 && (
                  <div className="flex items-center space-x-2 ml-auto">
                    <span className="text-sm text-muted-foreground">
                      {selectedAnomalies.size} selected
                    </span>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => handleBatchReview(true)}
                      disabled={isBatchProcessing}
                    >
                      <CheckSquare className="h-4 w-4 mr-1" />
                      Mark Reviewed
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => handleBatchReview(false)}
                      disabled={isBatchProcessing}
                    >
                      <Square className="h-4 w-4 mr-1" />
                      Mark Pending
                    </Button>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => setSelectedAnomalies(new Set())}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Anomalies Table */}
          <div className="border rounded-md overflow-hidden">
            <div className="w-full overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[50px]">
                      <Checkbox
                        checked={selectedAnomalies.size === anomalies.length && anomalies.length > 0}
                        onCheckedChange={handleSelectAll}
                      />
                    </TableHead>
                    <TableHead>Severity</TableHead>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>Source IP</TableHead>
                    <TableHead>Destination IP</TableHead>
                    <TableHead>Protocol</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Confidence</TableHead>
                    <TableHead>Attack Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {isLoading ? (
                    Array.from({ length: 5 }).map((_, index) => (
                      <TableRow key={index}>
                        {Array.from({ length: 10 }).map((_, cellIndex) => (
                          <TableCell key={cellIndex}>
                            <div className="h-6 bg-muted/30 animate-pulse rounded-md" />
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  ) : anomalies.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={10} className="text-center py-8">
                        <div className="flex flex-col items-center justify-center space-y-3">
                          <AlertTriangle className="h-8 w-8 text-muted-foreground" />
                          <p className="text-muted-foreground">No anomalies found</p>
                          {(severityFilter !== 'all' || attackTypeFilter !== 'all' || statusFilter !== 'all') && (
                            <Button 
                              variant="outline" 
                              size="sm" 
                              onClick={() => {
                                setSeverityFilter('all');
                                setAttackTypeFilter('all');
                                setStatusFilter('all');
                              }}
                            >
                              Clear all filters
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : (
                    anomalies.map((anomaly) => (
                      <TableRow key={anomaly.id}>
                        <TableCell>
                          <Checkbox
                            checked={selectedAnomalies.has(anomaly.id)}
                            onCheckedChange={(checked) => handleSelectAnomaly(anomaly.id, checked as boolean)}
                          />
                        </TableCell>
                        <TableCell>
                          <Badge className={getSeverityColor(anomaly.severity)} variant="outline">
                            {anomaly.severity.charAt(0).toUpperCase() + anomaly.severity.slice(1)}
                          </Badge>
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {formatDate(anomaly.timestamp)}
                        </TableCell>
                        <TableCell className="font-mono text-xs">{anomaly.sourceIp}</TableCell>
                        <TableCell className="font-mono text-xs">{anomaly.destinationIp}</TableCell>
                        <TableCell>{anomaly.protocol}</TableCell>
                        <TableCell>
                          <Badge variant="outline">
                            {anomaly.action}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="w-full bg-secondary rounded-full h-2.5">
                            <div
                              className="bg-cyberpurple h-2.5 rounded-full"
                              style={{ width: `${anomaly.confidence * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {(anomaly.confidence * 100).toFixed(0)}%
                          </span>
                        </TableCell>
                        <TableCell>
                          {anomaly.attackType ? (
                            <div className="space-y-1">
                              <Badge 
                                variant="outline" 
                                style={{ borderColor: anomaly.attackType.color, color: anomaly.attackType.color }}
                              >
                                {anomaly.attackType.name}
                              </Badge>
                              {anomaly.attackConfidence && (
                                <div className="text-xs text-muted-foreground">
                                  {(anomaly.attackConfidence * 100).toFixed(1)}% confidence
                                </div>
                              )}
                            </div>
                          ) : (
                            <span className="text-muted-foreground text-sm">N/A</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge variant={anomaly.reviewed ? "outline" : "secondary"}>
                            {anomaly.reviewed ? 'Reviewed' : 'Pending'}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Button asChild variant="ghost" size="sm">
                            <Link to={`/explanations/${anomaly.id}`}>Details</Link>
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-end space-x-2 py-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((old) => Math.max(old - 1, 1))}
                disabled={page === 1 || isLoading}
              >
                <ChevronLeft className="h-4 w-4" />
                <span className="sr-only">Previous Page</span>
              </Button>
              <div className="text-sm text-muted-foreground">
                Page {page} of {totalPages}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((old) => Math.min(old + 1, totalPages))}
                disabled={page === totalPages || isLoading}
              >
                <ChevronRight className="h-4 w-4" />
                <span className="sr-only">Next Page</span>
              </Button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default AnomaliesEnhanced;
