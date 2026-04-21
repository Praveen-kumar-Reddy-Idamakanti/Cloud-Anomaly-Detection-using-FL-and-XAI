# Tech Stack Details for Presentation

## 🏗️ **Architecture Overview**
**Cloud-Based Federated Anomaly Detection System with Explainable AI**

---

## 🔧 **Backend Technology Stack**

### **Core Framework & Language**
- **Python 3.8+** - Primary development language
- **FastAPI 0.104+** - High-performance async web framework
- **Uvicorn** - ASGI server for production deployment

### **Machine Learning & AI**
- **PyTorch 2.0+** - Deep learning framework
- **PyTorch Lightning** - Training orchestration
- **Scikit-learn 1.3+** - Traditional ML algorithms
- **NumPy 1.24+** - Numerical computing
- **Pandas 2.0+** - Data manipulation

### **Federated Learning**
- **Flower (FLWR) 1.7+** - Federated learning framework
- **Distributed training** across multiple clients
- **Privacy-preserving** model aggregation

### **Explainable AI (XAI)**
- **SHAP 0.44+** - SHapley Additive exPlanations
- **LIME 0.2+** - Local Interpretable Model-agnostic Explanations
- **Captum 0.6+** - PyTorch model interpretability
- **ELI5 0.13+** - Explain Like I'm 5
- **Interpret 0.2+** - Model interpretation toolkit
- **MLflow 1.26+** - Experiment tracking

### **Database & Storage**
- **SQLite 3.40+** - Lightweight database
- **MLflow Tracking** - Model versioning and experiments
- **File-based storage** for model artifacts

### **Data Visualization**
- **Matplotlib 3.8+** - Plotting library
- **Seaborn 0.13+** - Statistical visualization
- **Plotly 5.17+** - Interactive charts
- **tqdm 4.66+** - Progress bars

---

## 🎨 **Frontend Technology Stack**

### **Core Framework**
- **React 18.3.1** - Modern UI framework
- **TypeScript 5.5.3** - Type-safe JavaScript
- **Vite 5.4.1** - Fast build tool and dev server

### **UI Components & Styling**
- **Tailwind CSS 3.4.11** - Utility-first CSS framework
- **Radix UI** - Unstyled, accessible components
- **Lucide React 0.462.0** - Icon library
- **Framer Motion** - Animation library

### **State Management & Data**
- **TanStack React Query 5.56.2** - Server state management
- **React Hook Form 7.53.0** - Form handling
- **Zod 3.23.8** - Schema validation
- **Axios** - HTTP client

### **Charts & Visualization**
- **Recharts 2.12.7** - React chart library
- **D3.js** - Data visualization
- **Custom components** for anomaly detection plots

---

## 🗄️ **Data Processing Stack**

### **Data Sources**
- **CICIDS2017 Dataset** - Network intrusion detection
- **Real-time network traffic** - Live data ingestion
- **Federated client data** - Distributed datasets

### **Preprocessing Pipeline**
- **Feature engineering** - Network traffic features
- **Data normalization** - Min-max scaling
- **Label encoding** - Attack type categorization
- **Train/test splitting** - Stratified sampling

---

## 🚀 **Deployment & DevOps**

### **Containerization**
- **Docker** - Container orchestration
- **Docker Compose** - Multi-container deployment
- **Environment configuration** - .env management

### **Production Server**
- **Gunicorn 21.2+** - WSGI HTTP Server
- **Nginx** - Reverse proxy and load balancing
- **Process monitoring** - Psutil integration

### **Development Tools**
- **pytest 7.4+** - Testing framework
- **Black 23.0+** - Code formatting
- **Flake8 6.0+** - Linting
- **ESLint** - Frontend code quality

---

## 📊 **Model Architecture**

### **Two-Stage Classification**
1. **Stage 1**: Autoencoder for anomaly detection
   - Input: 79 network features
   - Architecture: 64→32→16→8→4 bottleneck
   - Output: Reconstruction error

2. **Stage 2**: Attack type classifier
   - Input: Same 79 features
   - Architecture: Multi-class neural network
   - Output: 5 attack categories (Botnet, DoS, Infiltration, Other, PortScan)

### **Model Performance**
- **Stage 1**: 68.3% F1-score, 67.5% ROC-AUC
- **Stage 2**: 92.7% accuracy, 54.3% F1-macro
- **Real-time inference** < 100ms per sample

---

## 🔐 **Security & Privacy**

### **Federated Learning Security**
- **Differential privacy** - Noise addition
- **Secure aggregation** - Encrypted model updates
- **Client isolation** - No raw data sharing

### **API Security**
- **JWT authentication** - Token-based auth
- **CORS configuration** - Cross-origin security
- **Input validation** - Pydantic schemas
- **Rate limiting** - DDoS protection

---

## 🌐 **Integration Capabilities**

### **API Endpoints**
- **RESTful API** - Standard HTTP methods
- **WebSocket** - Real-time updates
- **File upload** - Model deployment
- **Health checks** - System monitoring

### **Third-party Integrations**
- **SIEM systems** - Security event management
- **Network monitoring** - Traffic analysis tools
- **Logging services** - Centralized logging
- **Alert systems** - Notification services

---

## 📈 **Scalability Features**

### **Horizontal Scaling**
- **Load balancing** - Multiple server instances
- **Database sharding** - Distributed storage
- **Caching strategy** - Redis integration
- **CDN deployment** - Static asset delivery

### **Performance Optimization**
- **Async processing** - Non-blocking operations
- **Batch processing** - Efficient data handling
- **Model quantization** - Reduced memory usage
- **GPU acceleration** - CUDA support

---

## 🎯 **Key Differentiators**

1. **Federated Learning** - Privacy-preserving distributed training
2. **Explainable AI** - Model interpretability and transparency
3. **Two-Stage Architecture** - Anomaly detection + attack classification
4. **Real-time Processing** - Sub-second inference latency
5. **Modern Tech Stack** - Latest frameworks and best practices
6. **Comprehensive Monitoring** - Full observability stack

---

## 📋 **Technology Summary Table**

| Layer | Technology | Purpose | Version |
|-------|-------------|---------|---------|
| **Frontend** | React + TypeScript | UI Framework | 18.3.1 |
| **Styling** | Tailwind CSS + Radix UI | Component Library | 3.4.11 |
| **Backend** | FastAPI + Python | API Server | 0.104+ |
| **ML Framework** | PyTorch | Deep Learning | 2.0+ |
| **Federated** | Flower (FLWR) | Distributed Training | 1.7+ |
| **XAI** | SHAP + LIME + Captum | Model Interpretability | 0.44+ |
| **Database** | SQLite | Data Storage | 3.40+ |
| **Deployment** | Docker + Gunicorn | Production | Latest |
| **Monitoring** | MLflow + Custom | Experiment Tracking | 1.26+ |
