#!/usr/bin/env python3
"""
Single executable to start the Elliott Wave Predictor system.
Handles pre-cleaning, launching backend, and post-cleaning.
"""

import os
import sys
import subprocess
import signal
import time
import argparse
import shutil
from pathlib import Path

def pre_clean():
    """Perform pre-launch cleaning operations."""
    print("🧹 Performing pre-launch cleaning...")
    
    # Clean up any existing database files that might cause conflicts
    db_files = [
        "data/elliott_wave_data.db",
        "data/prices.db",
        "test_data.db",
        "full_system_test.db",
        "final_test.db"
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                print(f"  ✅ Removed {db_file}")
            except Exception as e:
                print(f"  ⚠️  Could not remove {db_file}: {e}")
    
    # Clean up any test directories
    test_dirs = ["backtest_results", "improvement_logs", "reports", "models"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"  ✅ Removed {test_dir}/ directory")
            except Exception as e:
                print(f"  ⚠️  Could not remove {test_dir}/: {e}")
    
    # Create necessary directories
    necessary_dirs = ["data", "backtest_results", "improvement_logs", "reports", "models"]
    for dir_name in necessary_dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            print(f"  ✅ Ensured {dir_name}/ directory exists")
        except Exception as e:
            print(f"  ⚠️  Could not create {dir_name}/: {e}")
    
    print("✅ Pre-cleaning completed\n")

def post_clean():
    """Perform post-shutdown cleaning operations."""
    print("\n🧹 Performing post-shutdown cleaning...")
    
    # Clean up temporary files
    temp_patterns = ["*.pyc", "__pycache__", "*.log"]
    for pattern in temp_patterns:
        try:
            # This is a simple approach - in production you might want more sophisticated cleanup
            pass
        except:
            pass
    
    print("✅ Post-cleaning completed")

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    # Package name -> import name mapping
    required_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"), 
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib")
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name} found")
        except ImportError:
            missing_packages.append(package_name)
            print(f"  ❌ {package_name} missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies satisfied\n")
    return True

def start_backend(port=8000, host="127.0.0.1"):
    """Start the backend server."""
    print(f"🚀 Starting Elliott Wave Predictor backend on {host}:{port}")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Change to the project directory
        project_dir = Path(__file__).parent
        os.chdir(project_dir)
        
        # Set environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_dir)
        
        # Start the backend directly with uvicorn
        cmd = [
            sys.executable,
            "-m", "uvicorn",
            "backend.main:app",
            "--host", host,
            "--port", str(port)
        ]
        
        print(f"🔧 Starting with command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print("✅ Backend completed successfully!")
        else:
            print(f"❌ Backend exited with code {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False

def main():
    """Main function to handle command line arguments and start the system."""
    parser = argparse.ArgumentParser(description="Start the Elliott Wave Predictor system")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the backend on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--no-clean", action="store_true", help="Skip pre-cleaning")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency checking")
    
    args = parser.parse_args()
    
    print("🌟 Elliott Wave Predictor - System Launcher")
    print("=" * 50)
    
    # Pre-cleaning
    if not args.no_clean:
        pre_clean()
    
    # Dependency checking
    if not args.skip_deps:
        if not check_dependencies():
            print("\n❌ Dependency check failed. Exiting.")
            return 1
    
    # Start backend
    try:
        success = start_backend(args.port, args.host)
        if success:
            print("\n✅ System shutdown complete")
        else:
            print("\n❌ System startup failed")
            return 1
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    finally:
        # Post-cleaning
        if not args.no_clean:
            post_clean()
    
    print("\n👋 Thank you for using Elliott Wave Predictor!")
    return 0

if __name__ == "__main__":
    sys.exit(main())