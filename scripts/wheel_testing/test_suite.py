#!/usr/bin/env python3
"""
Test suite script for cutde wheel testing.
This script runs the full pytest suite in an isolated environment.
"""

import sys
import os
import subprocess
import shutil
import tempfile
import traceback
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run cutde test suite")
    parser.add_argument("--tests-dir", required=True, help="Source tests directory")
    
    args = parser.parse_args()
    
    print(f"Starting test suite")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Tests source directory: {args.tests_dir}")
    
    temp_dir = None
    
    try:
        # Remove current directory from Python path to avoid local imports
        if "" in sys.path:
            sys.path.remove("")
        if "." in sys.path:
            sys.path.remove(".")
        
        print("Removed current directory from sys.path")
        
        # Copy tests to temp directory
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        
        tests_src = Path(args.tests_dir)
        tests_dest = Path(temp_dir) / "tests"
        
        print(f"Copying tests from {tests_src} to {tests_dest}")
        shutil.copytree(tests_src, tests_dest)
        
        # Change to temp directory and run pytest
        print(f"Changing to temporary directory: {temp_dir}")
        os.chdir(temp_dir)
        
        # Run pytest
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True)
        
        print("Test suite completed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"Test suite failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"ERROR in test suite: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temporary directory
        if temp_dir is not None:
            try:
                print(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

if __name__ == "__main__":
    main() 