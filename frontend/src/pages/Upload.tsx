import React, { useState } from 'react';
import { Upload } from 'lucide-react';
import Navbar from '../components/Layout/Navbar';
import Sidebar from '../components/Layout/Sidebar';
import LogUpload from '../components/Dashboard/LogUpload';

const UploadPage: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-4xl mx-auto">
          <div className="flex items-center mb-6">
            <Upload className="h-6 w-6 mr-2 text-cyberpurple" />
            <h1 className="text-2xl font-bold">Upload Log Files</h1>
          </div>

          <LogUpload />
        </div>
      </main>
    </div>
  );
};

export default UploadPage;
