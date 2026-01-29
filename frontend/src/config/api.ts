// API Configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  TIMEOUT: 30000, // Increased timeout for production
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
  MODEL_CONFIG: {
    INPUT_DIMENSIONS: 78,
    DEFAULT_THRESHOLD: 0.22610116,
    ATTACK_TYPES: ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris']
  }
};

// Environment configuration
export const ENV_CONFIG = {
  IS_DEVELOPMENT: import.meta.env.DEV,
  IS_PRODUCTION: import.meta.env.PROD,
  APP_NAME: import.meta.env.VITE_APP_NAME || 'Federated Anomaly Detection',
  APP_VERSION: import.meta.env.VITE_APP_VERSION || '1.0.0',
  ENVIRONMENT: import.meta.env.VITE_ENVIRONMENT || 'development',
};
