#!/usr/bin/env python3
"""
Test script to verify degenerate triangle handling in cutde.

Tests both fullspace and halfspace calculations with:
1. Normal triangles (should work as before) 
2. Degenerate triangles (should return zero without NaN)
"""

import numpy as np
import sys
import os

# Add the cutde directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cutde'))

try:
    import cutde.halfspace as halfspace
    import cutde.fullspace as fullspace
    print("✓ Successfully imported cutde modules")
except ImportError as e:
    print(f"✗ Failed to import cutde modules: {e}")
    print("Make sure cutde is properly built and installed")
    sys.exit(1)

def test_degenerate_triangles():
    """Test degenerate triangle handling."""
    print("\n=== Testing Degenerate Triangle Handling ===")
    
    # Halfspace constraint: z ≤ 0 for observation points
    obs_points = np.array([[0.5, 0.5, -1.0]])  # Note: z = -1.0
    slip = np.array([[1.0, 0.0, 0.0]])
    nu = 0.25
    
    test_cases = [
        ("Normal triangle", np.array([[[0,0,0], [1,0,0], [0,1,0]]])),
        ("All vertices coincide", np.array([[[0,0,0], [0,0,0], [0,0,0]]])),
        ("Collinear points", np.array([[[0,0,0], [1,0,0], [2,0,0]]])),
        ("Duplicate points", np.array([[[0,0,0], [0,0,0], [1,0,0]]])),
        ("Nearly degenerate", np.array([[[0,0,0], [np.finfo(float).eps,0,0], [0,np.finfo(float).eps,0]]])),
    ]
    
    print(f"Testing with observation point: {obs_points[0]}")
    print(f"Slip vector: {slip[0]}")
    print(f"Poisson's ratio: {nu}\n")
    
    all_passed = True
    
    for i, (description, triangles) in enumerate(test_cases):
        print(f"Test {i+1}: {description}")
        print(f"  Triangle vertices: {triangles[0]}")
        
        try:
            # Test halfspace displacement
            hs_disp = halfspace.disp(obs_points, triangles, slip, nu)
            print(f"  Halfspace displacement: {hs_disp[0]}")
            
            # Test fullspace displacement
            fs_disp = fullspace.disp(obs_points, triangles, slip, nu)
            print(f"  Fullspace displacement: {fs_disp[0]}")
            
            # Test halfspace strain
            hs_strain = halfspace.strain(obs_points, triangles, slip, nu)
            print(f"  Halfspace strain: {hs_strain[0]}")
            
            # Test fullspace strain  
            fs_strain = fullspace.strain(obs_points, triangles, slip, nu)
            print(f"  Fullspace strain: {fs_strain[0]}")
            
            # Validation
            if i == 0:  # Normal triangle
                if np.any(np.isnan(hs_disp)) or np.any(np.isnan(fs_disp)) or np.any(np.isnan(hs_strain)) or np.any(np.isnan(fs_strain)):
                    print("  ✗ FAIL: Normal triangle should not produce NaN")
                    all_passed = False
                elif np.allclose(hs_disp, 0.0) and np.allclose(fs_disp, 0.0):
                    print("  ✗ FAIL: Normal triangle should produce non-zero result")
                    all_passed = False
                else:
                    print("  ✓ PASS: Normal triangle produces valid non-zero results")
            else:  # Degenerate triangles
                if np.any(np.isnan(hs_disp)) or np.any(np.isnan(fs_disp)) or np.any(np.isnan(hs_strain)) or np.any(np.isnan(fs_strain)):
                    print("  ✗ FAIL: Degenerate triangle should not produce NaN")
                    all_passed = False
                elif not (np.allclose(hs_disp, 0.0) and np.allclose(fs_disp, 0.0) and np.allclose(hs_strain, 0.0) and np.allclose(fs_strain, 0.0)):
                    print("  ✗ FAIL: Degenerate triangle should produce zero results")
                    all_passed = False
                else:
                    print("  ✓ PASS: Degenerate triangle produces zero results without NaN")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False
        
        print()
    
    return all_passed

def main():
    """Main test function."""
    print("Testing cutde degenerate triangle handling")
    print("=" * 50)
    
    # Run the tests
    success = test_degenerate_triangles()
    
    print("=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Degenerate triangle handling is working correctly.")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("There are issues with degenerate triangle handling.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 