#!/usr/bin/env python3
"""
Comprehensive system connectivity check for backend-frontend-AI integration.
"""

import sys
import os
import json
import logging
from pathlib import Path
import requests
import sqlite3
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemConnectivityChecker:
    """Comprehensive system connectivity checker."""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5173"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
    
    def check_all(self):
        """Run all connectivity checks."""
        print("🔍 Starting Comprehensive System Connectivity Check")
        print("=" * 60)
        
        checks = [
            self.check_backend_api,
            self.check_ai_model_service,
            self.check_xai_integration,
            self.check_database_connectivity,
            self.check_model_artifacts,
            self.check_frontend_config,
            self.check_api_endpoints,
            self.check_real_time_features,
            self.check_end_to_end_flow
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                logger.error(f"Check failed: {e}")
                self.results["checks"][check.__name__] = {
                    "status": "ERROR",
                    "message": str(e)
                }
        
        self.generate_report()
        return self.results
    
    def check_backend_api(self):
        """Check backend API connectivity."""
        print("\n🔧 Checking Backend API...")
        
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.results["checks"]["check_backend_api"] = {
                    "status": "PASS",
                    "message": "Backend API is accessible",
                    "response_time": response.elapsed.total_seconds(),
                    "health_status": data.get("status", "unknown")
                }
                print(f"✅ Backend API: {data.get('status', 'OK')} ({response.elapsed.total_seconds():.2f}s)")
            else:
                self.results["checks"]["check_backend_api"] = {
                    "status": "FAIL",
                    "message": f"HTTP {response.status_code}"
                }
                print(f"❌ Backend API: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.results["checks"]["check_backend_api"] = {
                "status": "FAIL",
                "message": f"Connection failed: {e}"
            }
            print(f"❌ Backend API: Connection failed - {e}")
    
    def check_ai_model_service(self):
        """Check AI model service connectivity."""
        print("\n🤖 Checking AI Model Service...")
        
        try:
            response = requests.get(f"{self.backend_url}/model/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.results["checks"]["check_ai_model_service"] = {
                    "status": "PASS",
                    "message": "AI model service is accessible",
                    "model_loaded": data.get("status", "unknown"),
                    "input_dimensions": data.get("input_dim", "unknown"),
                    "two_stage_enabled": data.get("two_stage_enabled", False),
                    "attack_types": data.get("attack_types", [])
                }
                print(f"✅ AI Model Service: {data.get('status', 'OK')}")
                print(f"   - Input Dimensions: {data.get('input_dim', 'N/A')}")
                print(f"   - Two-Stage Enabled: {data.get('two_stage_enabled', 'False')}")
                print(f"   - Attack Types: {len(data.get('attack_types', []))}")
            else:
                self.results["checks"]["check_ai_model_service"] = {
                    "status": "FAIL",
                    "message": f"HTTP {response.status_code}"
                }
                print(f"❌ AI Model Service: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.results["checks"]["check_ai_model_service"] = {
                "status": "FAIL",
                "message": f"Connection failed: {e}"
            }
            print(f"❌ AI Model Service: Connection failed - {e}")
    
    def check_xai_integration(self):
        """Check XAI integration connectivity."""
        print("\n🧠 Checking XAI Integration...")
        
        try:
            # Test comprehensive explanation endpoint
            test_features = [0.1] * 78  # 78 features
            response = requests.post(
                f"{self.backend_url}/explain_anomaly",
                json={"features": test_features},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                self.results["checks"]["check_xai_integration"] = {
                    "status": "PASS",
                    "message": "XAI integration is working",
                    "explanation_type": data.get("explanation_type", "unknown"),
                    "has_phase1": "phase1" in data,
                    "has_phase2": "phase2" in data,
                    "has_phase3": "phase3" in data,
                    "comprehensive": data.get("comprehensive_explanation", False)
                }
                print(f"✅ XAI Integration: {data.get('explanation_type', 'OK')}")
                print(f"   - Phase 1: {'✅' if 'phase1' in data else '❌'}")
                print(f"   - Phase 2: {'✅' if 'phase2' in data else '❌'}")
                print(f"   - Phase 3: {'✅' if 'phase3' in data else '❌'}")
                print(f"   - Comprehensive: {'✅' if data.get('comprehensive_explanation') else '❌'}")
            else:
                self.results["checks"]["check_xai_integration"] = {
                    "status": "FAIL",
                    "message": f"HTTP {response.status_code}"
                }
                print(f"❌ XAI Integration: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.results["checks"]["check_xai_integration"] = {
                "status": "FAIL",
                "message": f"Connection failed: {e}"
            }
            print(f"❌ XAI Integration: Connection failed - {e}")
    
    def check_database_connectivity(self):
        """Check database connectivity."""
        print("\n💾 Checking Database Connectivity...")
        
        db_path = project_root / "data" / "anomaly_detection.db"
        
        if not db_path.exists():
            self.results["checks"]["check_database_connectivity"] = {
                "status": "FAIL",
                "message": "Database file not found"
            }
            print(f"❌ Database: File not found at {db_path}")
            return
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Check table counts
            table_counts = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                table_counts[table] = cursor.fetchone()[0]
            
            self.results["checks"]["check_database_connectivity"] = {
                "status": "PASS",
                "message": "Database is accessible",
                "database_path": str(db_path),
                "tables": tables,
                "table_counts": table_counts
            }
            
            print(f"✅ Database: Connected")
            print(f"   - Path: {db_path}")
            print(f"   - Tables: {len(tables)}")
            for table, count in table_counts.items():
                print(f"   - {table}: {count} records")
            
            conn.close()
            
        except Exception as e:
            self.results["checks"]["check_database_connectivity"] = {
                "status": "FAIL",
                "message": f"Database error: {e}"
            }
            print(f"❌ Database: Error - {e}")
    
    def check_model_artifacts(self):
        """Check model artifacts availability."""
        print("\n📁 Checking Model Artifacts...")
        
        artifacts_path = project_root / "model_artifacts"
        
        if not artifacts_path.exists():
            self.results["checks"]["check_model_artifacts"] = {
                "status": "FAIL",
                "message": "Model artifacts directory not found"
            }
            print(f"❌ Model Artifacts: Directory not found at {artifacts_path}")
            return
        
        artifacts = list(artifacts_path.glob("*.pth"))
        json_files = list(artifacts_path.glob("*.json"))
        
        self.results["checks"]["check_model_artifacts"] = {
            "status": "PASS",
            "message": "Model artifacts found",
            "artifacts_path": str(artifacts_path),
            "model_files": [f.name for f in artifacts],
            "config_files": [f.name for f in json_files],
            "total_files": len(artifacts) + len(json_files)
        }
        
        print(f"✅ Model Artifacts: Found {len(artifacts)} model files, {len(json_files)} config files")
        for artifact in artifacts[:5]:  # Show first 5
            print(f"   - {artifact.name} ({artifact.stat().size / 1024:.1f} KB)")
        
        if len(artifacts) > 5:
            print(f"   ... and {len(artifacts) - 5} more files")
    
    def check_frontend_config(self):
        """Check frontend configuration."""
        print("\n🎨 Checking Frontend Configuration...")
        
        frontend_path = project_root / "frontend"
        
        if not frontend_path.exists():
            self.results["checks"]["check_frontend_config"] = {
                "status": "FAIL",
                "message": "Frontend directory not found"
            }
            print(f"❌ Frontend: Directory not found at {frontend_path}")
            return
        
        # Check package.json
        package_json_path = frontend_path / "package.json"
        if package_json_path.exists():
            try:
                with open(package_json_path, 'r') as f:
                    package_data = json.load(f)
                
                self.results["checks"]["check_frontend_config"] = {
                    "status": "PASS",
                    "message": "Frontend configuration found",
                    "package_name": package_data.get("name", "unknown"),
                    "dependencies": len(package_data.get("dependencies", {})),
                    "scripts": package_data.get("scripts", {})
                }
                
                print(f"✅ Frontend Config: {package_data.get('name', 'unknown')}")
                print(f"   - Dependencies: {len(package_data.get('dependencies', {}))}")
                print(f"   - Scripts: {len(package_data.get('scripts', {}))}")
                
            except Exception as e:
                self.results["checks"]["check_frontend_config"] = {
                    "status": "FAIL",
                    "message": f"Error reading package.json: {e}"
                }
                print(f"❌ Frontend Config: Error reading package.json - {e}")
        else:
            self.results["checks"]["check_frontend_config"] = {
                "status": "FAIL",
                "message": "package.json not found"
            }
            print(f"❌ Frontend Config: package.json not found")
    
    def check_api_endpoints(self):
        """Check critical API endpoints."""
        print("\n🌐 Checking API Endpoints...")
        
        endpoints = [
            ("/health", "Health Check"),
            ("/model/info", "Model Info"),
            ("/stats", "System Stats"),
            ("/anomalies", "Anomalies"),
            ("/logs", "Logs"),
            ("/model/detect-enhanced", "Enhanced Detection")
        ]
        
        results = {}
        for endpoint, name in endpoints:
            try:
                if endpoint == "/model/detect-enhanced":
                    response = requests.post(
                        f"{self.backend_url}{endpoint}",
                        json={"features": [[0.1] * 78]},  # Test data
                        timeout=10
                    )
                else:
                    response = requests.get(f"{self.backend_url}{endpoint}", timeout=5)
                
                results[endpoint] = {
                    "status": "PASS" if response.status_code == 200 else "FAIL",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
                print(f"{'✅' if response.status_code == 200 else '❌'} {name}: HTTP {response.status_code} ({response.elapsed.total_seconds():.2f}s)")
                
            except Exception as e:
                results[endpoint] = {
                    "status": "FAIL",
                    "error": str(e)
                }
                print(f"❌ {name}: Error - {e}")
        
        self.results["checks"]["check_api_endpoints"] = {
            "status": "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL",
            "message": f"{sum(1 for r in results.values() if r['status'] == 'PASS')}/{len(results)} endpoints working",
            "endpoints": results
        }
    
    def check_real_time_features(self):
        """Check real-time features."""
        print("\n⚡ Checking Real-Time Features...")
        
        try:
            # Test SSE endpoint
            response = requests.get(f"{self.backend_url}/realtime/stream", timeout=5, stream=True)
            
            # Just check if the endpoint responds
            response.close()
            
            self.results["checks"]["check_real_time_features"] = {
                "status": "PASS",
                "message": "Real-time streaming endpoint is accessible"
            }
            print("✅ Real-Time Features: SSE endpoint accessible")
            
        except Exception as e:
            self.results["checks"]["check_real_time_features"] = {
                "status": "FAIL",
                "message": f"Real-time features error: {e}"
            }
            print(f"❌ Real-Time Features: Error - {e}")
    
    def check_end_to_end_flow(self):
        """Check end-to-end flow."""
        print("\n🔄 Checking End-to-End Flow...")
        
        try:
            # 1. Get model info
            model_response = requests.get(f"{self.backend_url}/model/info", timeout=10)
            if model_response.status_code != 200:
                raise Exception("Model info failed")
            
            model_data = model_response.json()
            
            # 2. Perform enhanced detection
            test_features = [0.1] * 78
            detection_response = requests.post(
                f"{self.backend_url}/model/detect-enhanced",
                json={"features": test_features},
                timeout=15
            )
            
            if detection_response.status_code != 200:
                raise Exception("Enhanced detection failed")
            
            detection_data = detection_response.json()
            
            # 3. Get XAI explanation
            xai_response = requests.post(
                f"{self.backend_url}/explain_anomaly",
                json={"features": test_features},
                timeout=15
            )
            
            if xai_response.status_code != 200:
                raise Exception("XAI explanation failed")
            
            xai_data = xai_response.json()
            
            # 4. Get system stats
            stats_response = requests.get(f"{self.backend_url}/stats", timeout=5)
            
            if stats_response.status_code != 200:
                raise Exception("System stats failed")
            
            stats_data = stats_response.json()
            
            self.results["checks"]["check_end_to_end_flow"] = {
                "status": "PASS",
                "message": "End-to-end flow is working",
                "model_status": model_data.get("status", "unknown"),
                "detection_success": len(detection_data.get("anomaly_predictions", [])) > 0,
                "xai_available": "explanation_type" in xai_data,
                "stats_available": "total_anomalies" in stats_data
            }
            
            print("✅ End-to-End Flow: Working")
            print(f"   - Model Status: {model_data.get('status', 'Unknown')}")
            print(f"   - Detection: {len(detection_data.get('anomaly_predictions', []))} predictions")
            print(f"   - XAI Available: {'✅' if 'explanation_type' in xai_data else '❌'}")
            print(f"   - Stats Available: {'✅' if 'total_anomalies' in stats_data else '❌'}")
            
        except Exception as e:
            self.results["checks"]["check_end_to_end_flow"] = {
                "status": "FAIL",
                "message": f"End-to-end flow failed: {e}"
            }
            print(f"❌ End-to-End Flow: Failed - {e}")
    
    def generate_report(self):
        """Generate comprehensive connectivity report."""
        print("\n" + "=" * 60)
        print("📊 CONNECTIVITY REPORT")
        print("=" * 60)
        
        total_checks = len(self.results["checks"])
        passed_checks = sum(1 for check in self.results["checks"].values() if check["status"] == "PASS")
        failed_checks = total_checks - passed_checks
        
        print(f"\n📈 Overall Status: {passed_checks}/{total_checks} checks passed")
        print(f"🎯 Success Rate: {(passed_checks/total_checks)*100:.1f}%")
        
        if failed_checks == 0:
            print("\n🎉 ALL SYSTEMS CONNECTED SUCCESSFULLY!")
            print("\n✅ Your backend-frontend-AI integration is fully operational.")
            print("✅ All components are correctly connected and functional.")
        else:
            print(f"\n⚠️  {failed_checks} systems need attention")
            print("\n❌ Issues found that need to be resolved:")
            
            for check_name, result in self.results["checks"].items():
                if result["status"] == "FAIL":
                    print(f"   - {check_name}: {result['message']}")
        
        # Save detailed report
        report_path = project_root / "system_connectivity_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        return passed_checks == total_checks

def main():
    """Main function to run connectivity checks."""
    checker = SystemConnectivityChecker()
    success = checker.check_all()
    return 0 if success else 1

if __name__ == "__main__":
    main()
