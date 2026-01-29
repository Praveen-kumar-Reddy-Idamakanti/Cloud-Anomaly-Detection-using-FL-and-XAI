
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

// Pages
import Index from "./pages/Index";
import Dashboard from "./pages/Dashboard";
import Anomalies from "./pages/Anomalies";
import Analytics from "./pages/Analytics";
import ModelManagement from "./pages/ModelManagement";
import Upload from "./pages/Upload";
import XAI from "./pages/XAI";
import XAIExplanation from "./pages/XAIExplanation";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/anomalies" element={<Anomalies />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/models" element={<ModelManagement />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/explanations" element={<XAI />} />
            <Route path="/explanations/:id" element={<XAIExplanation />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
