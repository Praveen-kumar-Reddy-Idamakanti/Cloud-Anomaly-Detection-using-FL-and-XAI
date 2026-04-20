
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, ChevronLeft, ChevronRight, Filter, RefreshCw, ChevronFirst, ChevronLast } from 'lucide-react';
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
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { anomaliesApi, AnomalyData } from '../api/api';

const Anomalies: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [anomalies, setAnomalies] = useState<AnomalyData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalAnomalies, setTotalAnomalies] = useState(0);
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [limit, setLimit] = useState(10);

  useEffect(() => {
    fetchAnomalies();
  }, [page, severityFilter, limit]);

  const fetchAnomalies = async () => {
    setIsLoading(true);
    try {
      const response = await anomaliesApi.getAnomalies(page, limit);
      
      // Apply severity filter if needed
      let filteredData = response.data;
      if (severityFilter !== 'all') {
        filteredData = response.data.filter(
          (anomaly) => anomaly.severity === severityFilter
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

  const handleLimitChange = (newLimit: number) => {
    setLimit(newLimit);
    setPage(1); // Reset to first page when changing items per page
  };

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const getSeverityColor = (severity: AnomalyData['severity']) => {
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

  // Pagination helper functions
  const getPageNumbers = () => {
    const delta = 2; // Number of pages to show on each side of current page
    const range = [];
    const rangeWithDots = [];
    let l;

    for (let i = 1; i <= totalPages; i++) {
      if (i == 1 || i == totalPages || (i >= page - delta && i <= page + delta)) {
        range.push(i);
      }
    }

    range.forEach((i) => {
      if (l) {
        if (i - l === 2) {
          rangeWithDots.push(l + 1);
        } else if (i - l !== 1) {
          rangeWithDots.push('...');
        }
      }
      rangeWithDots.push(i);
      l = i;
    });

    return rangeWithDots;
  };

  const handlePageJump = (pageNumber: number) => {
    if (pageNumber >= 1 && pageNumber <= totalPages) {
      setPage(pageNumber);
    }
  };

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <AlertTriangle className="h-6 w-6 mr-2 text-cyberpurple" />
              <h1 className="text-2xl font-bold">Anomalies</h1>
            </div>
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-2">
                <Filter className="h-4 w-4 text-muted-foreground" />
                <Select
                  value={severityFilter}
                  onValueChange={setSeverityFilter}
                >
                  <SelectTrigger className="w-[130px]">
                    <SelectValue placeholder="Filter by severity" />
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
              <div className="flex items-center space-x-2">
                <span className="text-sm text-muted-foreground">Show</span>
                <Select
                  value={limit.toString()}
                  onValueChange={(value) => handleLimitChange(Number(value))}
                >
                  <SelectTrigger className="w-[70px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="5">5</SelectItem>
                    <SelectItem value="10">10</SelectItem>
                    <SelectItem value="25">25</SelectItem>
                    <SelectItem value="50">50</SelectItem>
                    <SelectItem value="100">100</SelectItem>
                  </SelectContent>
                </Select>
                <span className="text-sm text-muted-foreground">entries</span>
              </div>
              <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isLoading}>
                <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>

          {/* Anomalies Table */}
          <div className="border rounded-md overflow-hidden">
            <div className="w-full overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Severity</TableHead>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>Source IP</TableHead>
                    <TableHead>Destination IP</TableHead>
                    <TableHead>Protocol</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Confidence</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {isLoading ? (
                    Array.from({ length: 5 }).map((_, index) => (
                      <TableRow key={index}>
                        {Array.from({ length: 9 }).map((_, cellIndex) => (
                          <TableCell key={cellIndex}>
                            <div className="h-6 bg-muted/30 animate-pulse rounded-md" />
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  ) : anomalies.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={9} className="text-center py-8">
                        <div className="flex flex-col items-center justify-center space-y-3">
                          <AlertTriangle className="h-8 w-8 text-muted-foreground" />
                          <p className="text-muted-foreground">No anomalies found</p>
                          {severityFilter !== 'all' && (
                            <Button variant="outline" size="sm" onClick={() => setSeverityFilter('all')}>
                              Clear filters
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : (
                    anomalies.map((anomaly) => (
                      <TableRow key={anomaly.id}>
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

          {/* Enhanced Pagination */}
          {totalPages > 1 && (
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 py-4 border-t">
              {/* Page Info */}
              <div className="text-sm text-muted-foreground">
                Showing {((page - 1) * limit) + 1} to {Math.min(page * limit, totalAnomalies)} of {totalAnomalies} entries
              </div>
              
              {/* Pagination Controls */}
              <div className="flex items-center space-x-1">
                {/* First Page */}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(1)}
                  disabled={page === 1 || isLoading}
                >
                  <ChevronFirst className="h-4 w-4" />
                  <span className="sr-only">First Page</span>
                </Button>
                
                {/* Previous Page */}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((old) => Math.max(old - 1, 1))}
                  disabled={page === 1 || isLoading}
                >
                  <ChevronLeft className="h-4 w-4" />
                  <span className="sr-only">Previous Page</span>
                </Button>
                
                {/* Page Numbers */}
                <div className="flex items-center space-x-1">
                  {getPageNumbers().map((pageNum, index) => (
                    <div key={index}>
                      {pageNum === '...' ? (
                        <span className="px-3 py-1 text-sm text-muted-foreground">...</span>
                      ) : (
                        <Button
                          variant={page === pageNum ? "default" : "outline"}
                          size="sm"
                          onClick={() => handlePageJump(pageNum as number)}
                          disabled={isLoading}
                          className="min-w-[40px]"
                        >
                          {pageNum}
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
                
                {/* Next Page */}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((old) => Math.min(old + 1, totalPages))}
                  disabled={page === totalPages || isLoading}
                >
                  <ChevronRight className="h-4 w-4" />
                  <span className="sr-only">Next Page</span>
                </Button>
                
                {/* Last Page */}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(totalPages)}
                  disabled={page === totalPages || isLoading}
                >
                  <ChevronLast className="h-4 w-4" />
                  <span className="sr-only">Last Page</span>
                </Button>
              </div>
              
              {/* Jump to Page */}
              <div className="flex items-center space-x-2">
                <span className="text-sm text-muted-foreground">Go to page</span>
                <Input
                  type="number"
                  min="1"
                  max={totalPages}
                  value={page}
                  onChange={(e) => {
                    const pageNum = parseInt(e.target.value);
                    if (!isNaN(pageNum)) {
                      handlePageJump(pageNum);
                    }
                  }}
                  className="w-16 h-8 text-center"
                  disabled={isLoading}
                />
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Anomalies;
