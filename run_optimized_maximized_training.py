#!/usr/bin/env python3
"""
Run maximized federated learning training with optimized settings for large datasets
"""
import subprocess
import time
import sys
import os
from pathlib import Path

def run_optimized_maximized_training():
    """Run the maximized federated learning training with optimizations"""
    print("🚀 OPTIMIZED MAXIMIZED FEDERATED LEARNING TRAINING")
    print("=" * 70)
    print("📊 Dataset: 1,622,672 samples across 8 clients")
    print("⚡ Optimizations: Increased timeout, faster training")
    print("🎯 Goal: Train precision-optimized anomaly detection")
    print("=" * 70)
    
    # Change to the correct directory
    fl_dir = Path("AI/federated_anomaly_detection")
    if not fl_dir.exists():
        print(f"❌ Directory not found: {fl_dir}")
        return 1
    
    os.chdir(fl_dir)
    
    try:
        # Start the optimized server with increased timeout
        print("\n🖥️  Starting OPTIMIZED server...")
        server_cmd = [
            "python", "optimized_server.py",
            "--input_dim", "79",
            "--min_clients", "8",  # 8 clients
            "--num_rounds", "5",   # 5 rounds
            "--address", "localhost:8080"
        ]
        
        print(f"🔧 Command: {' '.join(server_cmd)}")
        server_process = subprocess.Popen(server_cmd)
        
        # Give server time to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Start 8 clients with optimized settings
        client_processes = []
        for client_id in range(1, 9):  # Clients 1-8
            print(f"\n👤 Starting Optimized Client {client_id}...")
            client_cmd = [
                "python", "optimized_client.py",
                "--client-id", str(client_id),
                "--data-dir", "data/maximized",  # Use maximized data
                "--server-address", "localhost:8080"
            ]
            
            print(f"🔧 Command: {' '.join(client_cmd)}")
            client_process = subprocess.Popen(client_cmd)
            client_processes.append(client_process)
            
            # Stagger client starts to avoid overwhelming
            time.sleep(1)
        
        print(f"\n🎯 All 8 optimized clients started!")
        print("📊 Training with 1,622,672 total samples...")
        print("⚡ Optimizations applied:")
        print("   ⏱️  Increased timeout for large datasets")
        print("   🚀 Faster training parameters")
        print("   📈 Precision-optimized thresholds")
        
        # Wait for server to complete
        server_process.wait()
        
        print("\n🎉 OPTIMIZED MAXIMIZED TRAINING COMPLETED!")
        print("📊 Check 'checkpoints/' for saved models")
        print("📈 Training history available for analysis")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_optimized_maximized_training()
    sys.exit(exit_code)
