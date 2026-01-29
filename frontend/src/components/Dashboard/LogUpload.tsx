import React, { useState } from 'react';
import { Upload, Shield, Lock, AlertCircle, FileText, BarChart, Clock, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { toast } from 'sonner';
import { logsApi } from '../../api/api';

const LogUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [encrypted, setEncrypted] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [showUploadDialog, setShowUploadDialog] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setProgress(0);
    setShowUploadDialog(false);
    
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 10;
          if (newProgress >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return newProgress;
        });
      }, 300);
      
      await logsApi.uploadLog(file, encrypted);
      
      clearInterval(progressInterval);
      setProgress(100);
      
      // Add file to uploaded files
      setUploadedFiles(prev => [file.name, ...prev].slice(0, 5));
      
      setTimeout(() => {
        setUploading(false);
        setProgress(0);
        toast.success('Log file uploaded successfully');
        setFile(null);
        
        // After upload is complete, ask if user wants to analyze
        if (!analyzed) {
          toast('File uploaded', {
            description: 'Do you want to analyze this file for anomalies?',
            action: {
              label: 'Analyze',
              onClick: () => handleAnalyze(),
            },
          });
        }
      }, 500);
      
    } catch (error: any) {
      toast.error(error.message || 'Failed to upload log file');
      setUploading(false);
      setProgress(0);
    }
  };

  const handleUploadClick = () => {
    if (!file) return;
    setShowUploadDialog(true);
  };
  
  const handleAnalyze = async () => {
    setAnalyzing(true);
    
    try {
      // Simulate analysis processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      setAnalyzed(true);
      toast.success('Log file analyzed successfully', {
        description: 'New anomalies have been detected and added to your dashboard',
      });
    } catch (error: any) {
      toast.error(error.message || 'Failed to analyze log file');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <Card className="bg-gradient-to-br from-background to-secondary/30">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <Upload className="h-5 w-5 mr-2" /> Upload Log Files
        </CardTitle>
        <CardDescription>
          Upload your system logs for federated anomaly detection analysis. Files are {encrypted ? 'encrypted' : 'not encrypted'} during transfer.
        </CardDescription>
        
        <Alert className="mt-3 border-amber-200 bg-amber-50">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <AlertDescription className="text-amber-800">
            <strong>Research System:</strong> This federated learning system is for research/demonstration purposes. 
            Processing may take 5-15 minutes and is not suitable for production use.
          </AlertDescription>
        </Alert>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="upload" className="w-full">
          <TabsList className="mb-4">
            <TabsTrigger value="upload">Upload</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
          </TabsList>
          
          <TabsContent value="upload">
            <div
              className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                dragActive ? 'border-cyberpurple/70 bg-cyberpurple/5' : 'border-border'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {file ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-center">
                    <Shield className="h-8 w-8 text-cyberpurple" />
                  </div>
                  <p className="font-medium">{file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(file.size / 1024).toFixed(2)} KB
                  </p>
                  
                  {uploading ? (
                    <div className="space-y-2">
                      <Progress value={progress} className="h-2" />
                      <p className="text-sm text-muted-foreground">{progress}% uploaded</p>
                    </div>
                  ) : (
                    <Button onClick={handleUploadClick} className="mt-2">
                      {encrypted ? (
                        <>
                          <Lock className="h-4 w-4 mr-2" /> Encrypt & Upload
                        </>
                      ) : (
                        <>
                          <Upload className="h-4 w-4 mr-2" /> Upload
                        </>
                      )}
                    </Button>
                  )}
                  
                  {analyzing && (
                    <div className="mt-4">
                      <Progress value={undefined} className="h-2" />
                      <p className="text-sm text-muted-foreground mt-2">Analyzing log file...</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex justify-center">
                    <div className="h-16 w-16 rounded-full bg-muted flex items-center justify-center">
                      <Upload className="h-8 w-8 text-muted-foreground" />
                    </div>
                  </div>
                  <div>
                    <p className="font-medium mb-1">Drag and drop your log file</p>
                    <p className="text-sm text-muted-foreground mb-3">
                      or click to browse files
                    </p>
                    <input 
                      type="file" 
                      id="file-upload" 
                      className="hidden" 
                      onChange={handleFileChange} 
                    />
                    <Button asChild size="sm">
                      <label htmlFor="file-upload">Browse Files</label>
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="history">
            {uploadedFiles.length > 0 ? (
              <div className="space-y-2">
                <h3 className="text-sm font-medium mb-2">Recent Uploads</h3>
                {uploadedFiles.map((fileName, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-md">
                    <div className="flex items-center">
                      <FileText className="h-4 w-4 mr-2 text-muted-foreground" />
                      <span className="text-sm">{fileName}</span>
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => toast.info('Analysis opened')}>
                      <BarChart className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p>No upload history available</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
        
        <div className="flex items-center justify-between mt-4">
          <div className="flex items-center space-x-2">
            <Switch 
              id="encryption" 
              checked={encrypted} 
              onCheckedChange={setEncrypted}
            />
            <label 
              htmlFor="encryption" 
              className="text-sm font-medium cursor-pointer"
            >
              Enable encryption
            </label>
          </div>
          
          {!encrypted && (
            <div className="flex items-center text-amber-400 text-sm">
              <AlertCircle className="h-4 w-4 mr-1" />
              <span>Not recommended</span>
            </div>
          )}
        </div>
      </CardContent>

      {/* Upload Confirmation Dialog */}
      <Dialog open={showUploadDialog} onOpenChange={setShowUploadDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center">
              <AlertTriangle className="h-5 w-5 mr-2 text-amber-500" />
              Federated Anomaly Detection Processing
            </DialogTitle>
            <DialogDescription>
              Please read the following information before proceeding with the upload.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <Alert className="border-amber-200 bg-amber-50">
              <Clock className="h-4 w-4 text-amber-600" />
              <AlertDescription className="text-amber-800">
                <strong>Processing Time:</strong> Federated anomaly detection requires extensive processing across multiple distributed models. 
                This process may take <strong>5-15 minutes</strong> depending on file size and complexity.
              </AlertDescription>
            </Alert>

            <Alert className="border-red-200 bg-red-50">
              <AlertTriangle className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-800">
                <strong>Production Limitation:</strong> This is a research/demonstration system. 
                <strong>Not suitable for production environments</strong> due to:
                <ul className="mt-2 ml-4 list-disc space-y-1">
                  <li>Limited scalability and performance optimization</li>
                  <li>Research-grade model accuracy</li>
                  <li>No enterprise-grade security features</li>
                  <li>No SLA or uptime guarantees</li>
                </ul>
              </AlertDescription>
            </Alert>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-sm text-blue-800">
                <strong>What happens during processing:</strong>
              </p>
              <ul className="mt-1 ml-4 list-disc text-sm text-blue-700 space-y-1">
                <li>File encryption and secure transmission</li>
                <li>Distributed model inference across federated clients</li>
                <li>Anomaly detection and scoring</li>
                <li>XAI explanation generation</li>
                <li>Results aggregation and reporting</li>
              </ul>
            </div>
          </div>

          <DialogFooter className="flex-col sm:flex-row gap-2">
            <Button 
              variant="outline" 
              onClick={() => setShowUploadDialog(false)}
              className="w-full sm:w-auto"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleUpload}
              className="w-full sm:w-auto bg-cyberpurple hover:bg-cyberpurple/90"
            >
              <Upload className="h-4 w-4 mr-2" />
              Proceed with Upload
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
};

export default LogUpload;
