#!/usr/bin/env python3
"""
Database initialization script for SQLite database.
Run this script to initialize the database with sample data.
"""

import sys
import os

# Import path configuration
from config.app_config import path_config

from database.sqlite_setup import main

if __name__ == "__main__":
    print("🚀 Starting SQLite Database Initialization")
    print("=" * 50)
    
    try:
        exit_code = main()
        if exit_code == 0:
            print("\n✅ Database initialization completed successfully!")
            print("\n📋 Next steps:")
            print("1. Start the backend server: python -m uvicorn main:app --reload")
            print("2. Start the frontend: npm run dev")
            print("3. Access the application: http://localhost:5173")
        else:
            print("\n❌ Database initialization failed!")
            sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Database initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
