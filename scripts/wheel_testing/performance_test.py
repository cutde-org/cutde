#!/usr/bin/env python3
"""
Performance test script for cutde wheel testing.
This script is executed in isolated environments to test wheel performance.
"""

import sys
import platform
import time
import traceback
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Performance test for cutde")
    parser.add_argument(
        "--wheel-name", required=True, help="Name of the wheel being tested"
    )
    parser.add_argument("--target-python", required=True, help="Target Python version")
    parser.add_argument("--platform-info", required=True, help="Platform information")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save the plot"
    )

    args = parser.parse_args()

    print(f"Starting performance test for {args.wheel_name}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    try:
        # Remove current directory from Python path to avoid local imports
        if "" in sys.path:
            sys.path.remove("")
        if "." in sys.path:
            sys.path.remove(".")

        print("Removed current directory from sys.path")
        print(f"sys.path: {sys.path[:3]}...")  # Show first few entries

        print("Importing required modules...")
        import matplotlib

        matplotlib.use("Agg")
        print("matplotlib backend set to Agg")

        import matplotlib.pyplot as plt

        print("matplotlib.pyplot imported")

        import numpy as np

        print("numpy imported")

        import cutde.fullspace as FS

        print("cutde.fullspace imported")

        # Performance test setup
        print("Setting up performance test data...")
        xs = np.linspace(-2, 2, 1000)
        ys = np.linspace(-2, 2, 1000)
        obsx, obsy = np.meshgrid(xs, ys)
        pts = np.array([obsx, obsy, 0 * obsy]).reshape((3, -1)).T.copy()

        fault_pts = np.array([[-1, 0, 0], [1, 0, 0], [1, 0, -1], [-1, 0, -1]])
        fault_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        print(f"Test data shapes: pts={pts.shape}, fault_tris={fault_tris.shape}")

        # Time the displacement matrix computation
        print("Running displacement matrix computation...")
        start_time = time.time()
        disp_mat = FS.disp_matrix(obs_pts=pts, tris=fault_pts[fault_tris], nu=0.25)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Displacement matrix computation completed in {execution_time:.6f}s")
        print(f"disp_mat shape: {disp_mat.shape}")

        # Calculate displacement field
        print("Calculating displacement field...")
        slip = np.array([[1, 0, 0], [1, 0, 0]])
        disp = disp_mat.reshape((-1, 6)).dot(slip.flatten())
        disp_grid = disp.reshape(obsx.shape + (3,))

        print(f"disp_grid shape: {disp_grid.shape}")

        # Create visualization
        print("Creating visualization...")
        plt.figure(figsize=(12, 8), dpi=150)
        cntf = plt.contourf(obsx, obsy, disp_grid[:, :, 0], levels=21, cmap="viridis")
        plt.contour(
            obsx,
            obsy,
            disp_grid[:, :, 0],
            colors="k",
            linestyles="-",
            linewidths=0.5,
            levels=21,
        )
        plt.colorbar(cntf, label="Displacement ux")

        print("Plot creation completed")

        # Add detailed info to the plot
        python_info = f"Python {sys.version.split()[0]}"
        arch_info = platform.machine()

        plt.title(
            f"TDE Displacement Field\n"
            f"Platform: {args.platform_info} ({arch_info})\n"
            f"Wheel: {args.wheel_name}\n"
            f"Target: {args.target_python}, Actual: {python_info}\n"
            f"Execution Time: {execution_time:.4f}s"
        )
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.tight_layout()

        print("Plot formatting completed")

        # Save plot
        safe_wheel_name = args.wheel_name.replace("-", "_").replace(".", "_")
        output_filename = Path(args.output_dir) / f"tde_test_{safe_wheel_name}.png"
        output_filename.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_filename, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"Plot saved to: {output_filename}")

        # Output results in the expected format
        print(f"PERF_RESULTS:{execution_time:.6f}:{len(pts)}:{len(fault_tris)}")

        print("Performance test completed successfully")

    except Exception as e:
        print(f"ERROR in performance test: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
