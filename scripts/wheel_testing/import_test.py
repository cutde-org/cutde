#!/usr/bin/env python3
"""
Import test script for cutde wheel testing.
This script tests basic imports to ensure the wheel was installed correctly.
"""

import sys
import traceback

def main():
    print(f"Starting import test")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    try:
        # Remove current directory from Python path to avoid local imports
        if "" in sys.path:
            sys.path.remove("")
        if "." in sys.path:
            sys.path.remove(".")
        
        print("Removed current directory from sys.path")
        print(f"sys.path: {sys.path[:3]}...")  # Show first few entries
        
        # Test imports of installed wheel
        print("Testing cutde import...")
        import cutde
        print(f"cutde imported successfully from {cutde.__file__}")
        
        print("Testing cutde.fullspace import...")
        import cutde.fullspace
        print("cutde.fullspace imported successfully")
        
        print("Testing cutde.halfspace import...")
        import cutde.halfspace
        print("cutde.halfspace imported successfully")
        
        # Test basic functionality
        print("Testing basic functionality...")
        import numpy as np
        
        # Simple test to ensure the module works - use one observation point per triangle
        pts = np.array([[0, 0, 0]])  # 1 observation point
        tris = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])  # 1 triangle
        slips = np.array([[1, 0, 0]])  # 1 slip vector
        
        result = cutde.fullspace.disp(pts, tris, slips, 0.25)
        print(f"Basic displacement calculation successful: result shape = {result.shape}")
        
        print("All imports and basic functionality tests passed!")
        
    except Exception as e:
        print(f"ERROR in import test: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 