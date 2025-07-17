#!/usr/bin/env python3
"""
Comprehensive wheel testing script for cutde using micromamba for environment management.
Can be run locally or in CI/CD environments.

Enhanced debugging features:
- Use --verbose flag to see detailed debugging output
- Tracks which Python executable is found and used
- Shows decision process for environment selection
- Logs all command execution with clear status indicators

Example usage with debugging:
  python test_wheels.py --wheels-dir ./artifacts --results-dir ./results --verbose

This will show:
- Which Python executables are tested and found
- Version comparisons and compatibility checks
- Environment selection rationale (venv vs micromamba)
- Detailed command execution logs
- Wheel installation process details
"""

import argparse
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WheelTester:
    """Handles discovery, testing, and performance evaluation of wheels using micromamba."""
    
    def __init__(self, wheels_dir: Path, results_dir: Path, platform_name: Optional[str] = None):
        self.wheels_dir = Path(wheels_dir)
        self.results_dir = Path(results_dir)
        self.platform_name = platform_name or self._detect_platform()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._create_gitignore(self.results_dir)
        
        # Ensure wheels_dir exists and has .gitignore
        self.wheels_dir.mkdir(parents=True, exist_ok=True)
        self._create_gitignore(self.wheels_dir)
        
        self.force_micromamba = False
        self._check_micromamba()
    
    def _create_gitignore(self, directory: Path):
        """Create a .gitignore file in the specified directory that ignores all contents."""
        gitignore_path = directory / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write("*\n")
            logger.debug(f"Created .gitignore in {directory}")
        
    def _run_subprocess(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a subprocess with logging of the command."""
        cmd_str = ' '.join(shlex.quote(str(arg)) for arg in cmd)
        logger.info(f"Running: {cmd_str}")
        return subprocess.run(cmd, **kwargs)
    
    def _make_python_cmd(self, python_executable: str, args: List[str]) -> List[str]:
        """Create a subprocess command from a python executable and arguments.
        
        Handles cases where python_executable might be "py -3.11" (splits into separate args).
        """
        # Handle cases where python_executable might be "py -3.11" (split into separate args)
        python_args = python_executable.split() if ' ' in python_executable else [python_executable]
        return python_args + args
        
    def _check_micromamba(self):
        """Check if micromamba is available."""
        try:
            result = self._run_subprocess(['micromamba', '--version'], 
                                        capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            logger.info(f"micromamba {version} is available")
            self.micromamba_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("micromamba not found. Will use venv or warn if no compatible Python found.")
            self.micromamba_available = False
        
    def _detect_platform(self) -> str:
        """Detect the current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == "linux":
            # Detect musl vs glibc on Linux
            is_musl = self._is_musl_system()
            
            if "aarch64" in machine or "arm64" in machine:
                return "linux-aarch64-musl" if is_musl else "linux-aarch64"
            else:
                return "linux-x86_64-musl" if is_musl else "linux-x86_64"
        elif system == "darwin":
            if "arm64" in machine:
                return "macos-arm64"
            else:
                return "macos-x86_64"
        elif system == "windows":
            return "windows-amd64"
        else:
            return f"{system}-{machine}"
    
    def _is_musl_system(self) -> bool:
        """Check if the system is using musl libc."""
        try:
            # Try to run ldd --version to check for musl
            result = self._run_subprocess(['ldd', '--version'], capture_output=True, text=True)
            output = result.stderr + result.stdout
            
            if 'musl' in output.lower():
                return True
            elif 'glibc' in output.lower() or 'gnu libc' in output.lower():
                return False
            else:
                # Fallback: check if we can find musl in common locations
                musl_paths = ['/lib/ld-musl-*.so*', '/usr/lib/libc.musl-*.so*']
                for path_pattern in musl_paths:
                    if self._run_subprocess(['sh', '-c', f'ls {path_pattern} 2>/dev/null'], 
                                          capture_output=True).returncode == 0:
                        return True
                return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def discover_wheels(self) -> List[Tuple[Path, str]]:
        """Discover wheels compatible with the current platform."""
        logger.info(f"Discovering wheels for platform: {self.platform_name}")
        
        # Define platform patterns
        # Note: manylinux wheels are for glibc-based systems, musllinux wheels are for musl-based systems
        platform_patterns = {
            "linux-x86_64": [r"linux_x86_64", r"manylinux.*x86_64"],  # glibc systems
            "linux-x86_64-musl": [r"linux_x86_64", r"musllinux.*x86_64"],  # musl systems (Alpine, etc.)
            "linux-aarch64": [r"linux_aarch64", r"manylinux.*aarch64"],  # glibc systems
            "linux-aarch64-musl": [r"linux_aarch64", r"musllinux.*aarch64"],  # musl systems
            "windows-amd64": [r"win_amd64", r"win32"],
            "macos-x86_64": [r"macosx.*x86_64"],
            "macos-arm64": [r"macosx.*arm64", r"macosx.*universal2"],
        }
        
        patterns = platform_patterns.get(self.platform_name, [])
        if not patterns:
            logger.warning(f"No patterns defined for platform: {self.platform_name}")
            return []
        
        wheels = []
        for wheel_file in self.wheels_dir.glob("*.whl"):
            for pattern in patterns:
                if re.search(pattern, wheel_file.name):
                    # Extract Python version from wheel filename
                    python_match = re.search(r"cp(\d+)", wheel_file.name)
                    if python_match:
                        python_ver = python_match.group(1)
                        major = python_ver[0]
                        minor = python_ver[1:] if len(python_ver) > 1 else "0"
                        python_version = f"{major}.{minor}"
                        wheels.append((wheel_file, python_version))
                        logger.info(f"Found wheel: {wheel_file.name} (Python {python_version})")
                    break
        
        logger.info(f"Discovered {len(wheels)} compatible wheels")
        return wheels
    
    def _create_micromamba_env(self, wheel_path: Path, python_version: str) -> str:
        """Create a micromamba environment for testing a wheel."""
        wheel_name = wheel_path.stem.replace("-", "_").replace(".", "_")
        env_name = f"test_{wheel_name}"
        
        logger.info(f"Creating micromamba environment: {env_name} (Python {python_version})")
        
        try:
            # Create environment with specific Python version
            self._run_subprocess([
                'micromamba', 'create', '-n', env_name, 
                f'python={python_version}', 
                'pip', 'pytest', 'numpy', 'scipy', 'matplotlib', 'pyproj',
                '-c', 'conda-forge', 
                '--yes'
            ], check=True, capture_output=True)
            
            logger.info(f"Created environment: {env_name}")
            return env_name
            
        except subprocess.CalledProcessError as e:
            # If exact version fails, try with just major.minor
            logger.warning(f"Failed to create environment with Python {python_version}, trying latest {python_version.split('.')[0]}.x")
            try:
                major_version = python_version.split('.')[0]
                self._run_subprocess([
                    'micromamba', 'create', '-n', env_name, 
                    f'python={major_version}', 
                    'pip', 'pytest', 'numpy', 'scipy', 'matplotlib', 'pyproj',
                    '-c', 'conda-forge', 
                    '--yes'
                ], check=True, capture_output=True)
                
                logger.info(f"Created environment: {env_name} with Python {major_version}.x")
                return env_name
                
            except subprocess.CalledProcessError:
                logger.error(f"Failed to create micromamba environment: {e}")
                raise
    
    def _create_venv(self, wheel_path: Path, python_version: str) -> Path:
        """Fallback: Create a virtual environment for testing a wheel."""
        wheel_name = wheel_path.stem
        venv_dir = self.results_dir / f"venv_{wheel_name}"
        
        # Try to find the specific Python version
        python_cmd = self._find_python_executable(python_version)
        
        logger.info(f"Creating virtual environment:")
        logger.info(f"  Target Python version: {python_version}")
        logger.info(f"  Selected executable: {python_cmd}")
        logger.info(f"  Virtual environment directory: {venv_dir}")
        
        # Handle existing venv directory
        if venv_dir.exists():
            logger.info(f"  Existing venv directory found, removing: {venv_dir}")
            try:
                shutil.rmtree(venv_dir)
                logger.info(f"  ✅ Removed existing venv directory")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not remove existing venv directory: {e}")
                logger.info(f"  Trying to continue anyway...")
        
        # Verify the executable version before creating venv
        actual_version = None
        try:
            result = self._run_subprocess(self._make_python_cmd(python_cmd, ["--version"]), 
                                        capture_output=True, text=True, check=True)
            actual_version = result.stdout.strip()
            logger.info(f"  Verified executable version: {actual_version}")
        except Exception as e:
            logger.warning(f"  Could not verify executable version: {e}")
        
        # Create the venv with --copies to avoid symlink issues
        venv_cmd = self._make_python_cmd(python_cmd, ["-m", "venv", "--copies", str(venv_dir)])
        logger.info(f"  Command: {' '.join(venv_cmd)}")
        logger.info(f"  Using --copies flag to avoid symlink issues")
        
        self._run_subprocess(venv_cmd, check=True)
        logger.info(f"  ✅ Virtual environment created successfully")
        
        # Verify that the venv actually contains the correct Python version
        venv_python = self._get_venv_python(venv_dir)
        try:
            result = self._run_subprocess(self._make_python_cmd(venv_python, ["--version"]), 
                                        capture_output=True, text=True, check=True)
            venv_version = result.stdout.strip()
            logger.info(f"  Verified venv Python version: {venv_version}")
            
            # Check if it matches what we expected
            if actual_version and "Python" in actual_version and "Python" in venv_version:
                expected_version = actual_version.split()[1]
                actual_venv_version = venv_version.split()[1]
                expected_major_minor = ".".join(expected_version.split(".")[:2])
                actual_major_minor = ".".join(actual_venv_version.split(".")[:2])
                
                if expected_major_minor == actual_major_minor:
                    logger.info(f"  ✅ Venv Python version matches: {expected_major_minor}")
                else:
                    logger.error(f"  ❌ Venv Python version mismatch: expected {expected_major_minor}, got {actual_major_minor}")
                    logger.error(f"  This may cause wheel compatibility issues!")
            
        except Exception as e:
            logger.warning(f"  Could not verify venv Python version: {e}")
        
        return venv_dir
    
    def _find_python_executable(self, target_version: str) -> str:
        """Find the best Python executable for the target version."""
        logger.info(f"Finding Python executable for version {target_version}")
        
        # First, check if we already found a compatible executable during _can_use_venv
        if hasattr(self, '_compatible_python_executable') and hasattr(self, '_compatible_python_version'):
            logger.info(f"Using pre-validated executable: {self._compatible_python_executable} ({self._compatible_python_version})")
            # Clear the cached values to avoid reuse
            executable = self._compatible_python_executable
            version = self._compatible_python_version
            delattr(self, '_compatible_python_executable')
            delattr(self, '_compatible_python_version')
            return executable
        
        # Windows-specific handling
        if sys.platform == "win32":
            return self._find_python_executable_windows(target_version)
        
        # Unix-like systems (Linux, macOS)
        candidates = [
            f"python{target_version}",
            f"python{target_version.split('.')[0]}",
            "python3",
            "python"
        ]
        
        logger.info(f"Searching for Python executable among candidates: {candidates}")
        
        for candidate in candidates:
            try:
                logger.debug(f"Testing candidate: {candidate}")
                result = self._run_subprocess([candidate, "--version"], 
                                            capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    logger.info(f"Found Python executable: {candidate} -> {version}")
                    
                    # Get the full path to the executable
                    try:
                        which_result = self._run_subprocess(["which", candidate], 
                                                          capture_output=True, text=True, check=True)
                        full_path = which_result.stdout.strip()
                        logger.info(f"Full path: {full_path}")
                        
                        # Parse version to check compatibility
                        if "Python" in version:
                            actual_version = version.split()[1]
                            actual_major_minor = ".".join(actual_version.split(".")[:2])
                            target_major_minor = ".".join(target_version.split(".")[:2])
                            
                            if actual_major_minor == target_major_minor:
                                logger.info(f"✅ Selected compatible executable: {full_path} (exact match)")
                            else:
                                logger.warning(f"⚠️ Version mismatch: {full_path} has {actual_major_minor}, target is {target_major_minor}")
                                logger.warning(f"Using {full_path} anyway (fallback behavior)")
                        
                        return full_path
                    except subprocess.CalledProcessError:
                        logger.warning(f"Could not get full path for {candidate}, using command name")
                        return candidate
                else:
                    logger.debug(f"❌ {candidate} returned non-zero exit code: {result.returncode}")
            except FileNotFoundError:
                logger.debug(f"❌ {candidate} not found")
                continue
        
        logger.error(f"No suitable Python executable found for version {target_version}")
        raise RuntimeError("No suitable Python executable found")
    
    def _find_python_executable_windows(self, target_version: str) -> str:
        """Find Python executable on Windows using py launcher or hostedtoolcache."""
        logger.info(f"Finding Python executable on Windows for version {target_version}")
        
        # Try py launcher first (most reliable on Windows)
        try:
            py_version_arg = f"-{target_version}"
            logger.debug(f"Testing py launcher: py {py_version_arg}")
            result = self._run_subprocess(["py", py_version_arg, "--version"], 
                                        capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Found Python via py launcher: py {py_version_arg} -> {version}")
                
                # Verify version matches
                if "Python" in version:
                    actual_version = version.split()[1]
                    actual_major_minor = ".".join(actual_version.split(".")[:2])
                    target_major_minor = ".".join(target_version.split(".")[:2])
                    
                    if actual_major_minor == target_major_minor:
                        logger.info(f"✅ Selected compatible executable: py {py_version_arg} (exact match)")
                        return f"py {py_version_arg}"
                    else:
                        logger.debug(f"❌ Version mismatch: py {py_version_arg} has {actual_major_minor}, need {target_major_minor}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug(f"py launcher not available or failed for {target_version}")
        
        # Try to find Python in hostedtoolcache (GitHub Actions)
        hostedtoolcache_paths = [
            Path(f"C:/hostedtoolcache/windows/Python/{target_version}.*/x64/python.exe"),
            Path(f"C:/hostedtoolcache/windows/Python/{target_version}/x64/python.exe"),
        ]
        
        for path_pattern in hostedtoolcache_paths:
            try:
                # Use glob to find matching paths
                import glob
                matches = glob.glob(str(path_pattern))
                if matches:
                    # Sort to get the latest version
                    matches.sort(reverse=True)
                    python_exe = matches[0]
                    logger.debug(f"Testing hostedtoolcache path: {python_exe}")
                    
                    # Test the executable
                    result = self._run_subprocess([python_exe, "--version"], 
                                                capture_output=True, text=True)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        logger.info(f"Found Python in hostedtoolcache: {python_exe} -> {version}")
                        
                        # Verify version matches
                        if "Python" in version:
                            actual_version = version.split()[1]
                            actual_major_minor = ".".join(actual_version.split(".")[:2])
                            target_major_minor = ".".join(target_version.split(".")[:2])
                            
                            if actual_major_minor == target_major_minor:
                                logger.info(f"✅ Selected compatible executable: {python_exe} (exact match)")
                                return python_exe
                            else:
                                logger.debug(f"❌ Version mismatch: {python_exe} has {actual_major_minor}, need {target_major_minor}")
            except Exception as e:
                logger.debug(f"Error searching hostedtoolcache pattern {path_pattern}: {e}")
        
        # Try standard Windows Python commands
        candidates = ["python", "python3"]
        for candidate in candidates:
            try:
                logger.debug(f"Testing candidate: {candidate}")
                result = self._run_subprocess([candidate, "--version"], 
                                            capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    logger.info(f"Found Python executable: {candidate} -> {version}")
                    
                    # Parse version to check compatibility
                    if "Python" in version:
                        actual_version = version.split()[1]
                        actual_major_minor = ".".join(actual_version.split(".")[:2])
                        target_major_minor = ".".join(target_version.split(".")[:2])
                        
                        if actual_major_minor == target_major_minor:
                            logger.info(f"✅ Selected compatible executable: {candidate} (exact match)")
                            return candidate
                        else:
                            logger.debug(f"❌ Version mismatch: {candidate} has {actual_major_minor}, need {target_major_minor}")
                else:
                    logger.debug(f"❌ {candidate} returned non-zero exit code: {result.returncode}")
            except FileNotFoundError:
                logger.debug(f"❌ {candidate} not found")
                continue
        
        logger.error(f"No suitable Python {target_version} executable found on Windows")
        raise RuntimeError(f"No suitable Python {target_version} executable found on Windows")
    
    def _can_use_venv(self, target_version: str) -> bool:
        """Check if we can use venv with a compatible Python version."""
        logger.info(f"Checking for compatible Python {target_version} executable for venv...")
        
        # Windows-specific handling
        if sys.platform == "win32":
            return self._can_use_venv_windows(target_version)
        
        # Unix-like systems (Linux, macOS)
        candidates = [
            f"python{target_version}",
            f"python{target_version.split('.')[0]}",
            "python3",
            "python"
        ]
        
        for candidate in candidates:
            try:
                logger.debug(f"Testing candidate: {candidate}")
                result = self._run_subprocess([candidate, "--version"], 
                                            capture_output=True, text=True)
                if result.returncode == 0:
                    # Extract actual version from output
                    version_output = result.stdout.strip()
                    logger.debug(f"Found executable {candidate} -> {version_output}")
                    
                    # Parse version like "Python 3.13.5" -> "3.13"
                    if "Python" in version_output:
                        actual_version = version_output.split()[1]
                        actual_major_minor = ".".join(actual_version.split(".")[:2])
                        target_major_minor = ".".join(target_version.split(".")[:2])
                        
                        logger.debug(f"Version comparison: {actual_major_minor} vs {target_major_minor}")
                        
                        # Only return True if the actual version matches the target
                        if actual_major_minor == target_major_minor:
                            # Get the full path to the executable
                            try:
                                which_result = self._run_subprocess(["which", candidate], 
                                                                  capture_output=True, text=True, check=True)
                                full_path = which_result.stdout.strip()
                                logger.info(f"✅ Found compatible Python: {full_path} ({version_output}) matches target {target_version}")
                                # Store the found executable full path for consistency
                                self._compatible_python_executable = full_path
                                self._compatible_python_version = version_output
                                return True
                            except subprocess.CalledProcessError:
                                logger.warning(f"Could not get full path for {candidate}, storing command name")
                                logger.info(f"✅ Found compatible Python: {candidate} ({version_output}) matches target {target_version}")
                                self._compatible_python_executable = candidate
                                self._compatible_python_version = version_output
                                return True
                        else:
                            logger.debug(f"❌ Version mismatch: {candidate} has {actual_major_minor}, need {target_major_minor}")
                else:
                    logger.debug(f"❌ {candidate} returned non-zero exit code: {result.returncode}")
            except FileNotFoundError:
                logger.debug(f"❌ {candidate} not found")
                continue
        
        logger.info(f"❌ No compatible Python {target_version} executable found for venv")
        return False
    
    def _can_use_venv_windows(self, target_version: str) -> bool:
        """Check if we can use venv with a compatible Python version on Windows."""
        logger.info(f"Checking for compatible Python {target_version} executable for venv on Windows...")
        
        # Try py launcher first (most reliable on Windows)
        try:
            py_version_arg = f"-{target_version}"
            logger.debug(f"Testing py launcher: py {py_version_arg}")
            result = self._run_subprocess(["py", py_version_arg, "--version"], 
                                        capture_output=True, text=True)
            if result.returncode == 0:
                version_output = result.stdout.strip()
                logger.debug(f"Found executable py {py_version_arg} -> {version_output}")
                
                # Parse version like "Python 3.13.5" -> "3.13"
                if "Python" in version_output:
                    actual_version = version_output.split()[1]
                    actual_major_minor = ".".join(actual_version.split(".")[:2])
                    target_major_minor = ".".join(target_version.split(".")[:2])
                    
                    logger.debug(f"Version comparison: {actual_major_minor} vs {target_major_minor}")
                    
                    if actual_major_minor == target_major_minor:
                        logger.info(f"✅ Found compatible Python: py {py_version_arg} ({version_output}) matches target {target_version}")
                        self._compatible_python_executable = f"py {py_version_arg}"
                        self._compatible_python_version = version_output
                        return True
                    else:
                        logger.debug(f"❌ Version mismatch: py {py_version_arg} has {actual_major_minor}, need {target_major_minor}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug(f"py launcher not available or failed for {target_version}")
        
        # Try to find Python in hostedtoolcache (GitHub Actions)
        hostedtoolcache_paths = [
            Path(f"C:/hostedtoolcache/windows/Python/{target_version}.*/x64/python.exe"),
            Path(f"C:/hostedtoolcache/windows/Python/{target_version}/x64/python.exe"),
        ]
        
        for path_pattern in hostedtoolcache_paths:
            try:
                # Use glob to find matching paths
                import glob
                matches = glob.glob(str(path_pattern))
                if matches:
                    # Sort to get the latest version
                    matches.sort(reverse=True)
                    python_exe = matches[0]
                    logger.debug(f"Testing hostedtoolcache path: {python_exe}")
                    
                    # Test the executable
                    result = self._run_subprocess([python_exe, "--version"], 
                                                capture_output=True, text=True)
                    if result.returncode == 0:
                        version_output = result.stdout.strip()
                        logger.debug(f"Found executable {python_exe} -> {version_output}")
                        
                        # Parse version like "Python 3.13.5" -> "3.13"
                        if "Python" in version_output:
                            actual_version = version_output.split()[1]
                            actual_major_minor = ".".join(actual_version.split(".")[:2])
                            target_major_minor = ".".join(target_version.split(".")[:2])
                            
                            logger.debug(f"Version comparison: {actual_major_minor} vs {target_major_minor}")
                            
                            if actual_major_minor == target_major_minor:
                                logger.info(f"✅ Found compatible Python: {python_exe} ({version_output}) matches target {target_version}")
                                self._compatible_python_executable = python_exe
                                self._compatible_python_version = version_output
                                return True
                            else:
                                logger.debug(f"❌ Version mismatch: {python_exe} has {actual_major_minor}, need {target_major_minor}")
            except Exception as e:
                logger.debug(f"Error searching hostedtoolcache pattern {path_pattern}: {e}")
        
        # Try standard Windows Python commands
        candidates = ["python", "python3"]
        for candidate in candidates:
            try:
                logger.debug(f"Testing candidate: {candidate}")
                result = self._run_subprocess([candidate, "--version"], 
                                            capture_output=True, text=True)
                if result.returncode == 0:
                    version_output = result.stdout.strip()
                    logger.debug(f"Found executable {candidate} -> {version_output}")
                    
                    # Parse version like "Python 3.13.5" -> "3.13"
                    if "Python" in version_output:
                        actual_version = version_output.split()[1]
                        actual_major_minor = ".".join(actual_version.split(".")[:2])
                        target_major_minor = ".".join(target_version.split(".")[:2])
                        
                        logger.debug(f"Version comparison: {actual_major_minor} vs {target_major_minor}")
                        
                        if actual_major_minor == target_major_minor:
                            logger.info(f"✅ Found compatible Python: {candidate} ({version_output}) matches target {target_version}")
                            self._compatible_python_executable = candidate
                            self._compatible_python_version = version_output
                            return True
                        else:
                            logger.debug(f"❌ Version mismatch: {candidate} has {actual_major_minor}, need {target_major_minor}")
                else:
                    logger.debug(f"❌ {candidate} returned non-zero exit code: {result.returncode}")
            except FileNotFoundError:
                logger.debug(f"❌ {candidate} not found")
                continue
        
        logger.info(f"❌ No compatible Python {target_version} executable found for venv on Windows")
        return False
    
    def _run_in_micromamba_env(self, env_name: str, command: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a command in a micromamba environment."""
        # Use micromamba run to execute commands in the environment
        full_command = ['micromamba', 'run', '-n', env_name] + command
        return self._run_subprocess(full_command, **kwargs)
    
    def _get_venv_python(self, venv_dir: Path) -> str:
        """Get the Python executable path for a virtual environment (fallback)."""
        if sys.platform == "win32":
            return str(venv_dir / "Scripts" / "python.exe")
        else:
            return str(venv_dir / "bin" / "python")
    

    
    def test_wheel(self, wheel_path: Path, python_version: str) -> Dict:
        """Test a single wheel and return results."""
        logger.info(f"Testing wheel: {wheel_path.name} (target Python {python_version})")
        
        # Force micromamba if requested
        if self.force_micromamba:
            if self.micromamba_available:
                logger.info(f"Forcing use of micromamba for Python {python_version}")
                return self._test_wheel_micromamba(wheel_path, python_version)
            else:
                error_msg = "Micromamba forced but not available"
                logger.error(error_msg)
                return {
                    "wheel": wheel_path.name,
                    "target_python": python_version,
                    "actual_python": "N/A",
                    "platform": self.platform_name,
                    "architecture": platform.machine(),
                    "status": "FAILED",
                    "environment_type": "none",
                    "error": error_msg
                }
        
        # Decision process logging
        logger.info("🔍 Determining environment strategy...")
        logger.info(f"Available options: micromamba={self.micromamba_available}, venv=(checking...)")
        
        # Prefer venv if compatible Python version is available
        can_use_venv = self._can_use_venv(python_version)
        
        # Log the decision rationale
        if can_use_venv:
            logger.info(f"✅ Decision: Using venv (compatible Python {python_version} found)")
            compatible_exec = getattr(self, '_compatible_python_executable', 'unknown')
            compatible_ver = getattr(self, '_compatible_python_version', 'unknown')
            logger.info(f"   Executable: {compatible_exec} -> {compatible_ver}")
            return self._test_wheel_venv(wheel_path, python_version)
        elif self.micromamba_available:
            logger.info(f"✅ Decision: Using micromamba (no compatible Python {python_version} found locally)")
            logger.info(f"   micromamba can create Python {python_version} environment")
            return self._test_wheel_micromamba(wheel_path, python_version)
        else:
            error_msg = f"No compatible Python {python_version} found and micromamba is not available"
            logger.error(f"❌ Decision: Cannot test wheel - {error_msg}")
            return {
                "wheel": wheel_path.name,
                "target_python": python_version,
                "actual_python": "N/A",
                "platform": self.platform_name,
                "architecture": platform.machine(),
                "status": "FAILED",
                "environment_type": "none",
                "error": error_msg
            }
    
    def _test_wheel_micromamba(self, wheel_path: Path, python_version: str) -> Dict:
        """Test a wheel using micromamba environment."""
        env_name = self._create_micromamba_env(wheel_path, python_version)
        
        try:
            # Get actual Python version in environment
            result = self._run_in_micromamba_env(env_name, ["python", "--version"], 
                                               capture_output=True, text=True, check=True)
            actual_python = result.stdout.strip().split()[1]
            
            # Install the wheel
            logger.info("Installing wheel...")
            self._run_in_micromamba_env(env_name, ["python", "-m", "pip", "install", str(wheel_path)], check=True)
            
            # Run import tests
            logger.info("Running import tests...")
            self._run_import_tests_micromamba(env_name)
            
            # Run test suite
            logger.info("Running test suite...")
            self._run_test_suite_micromamba(env_name)
            
            # Generate performance visualization
            logger.info("Generating performance visualization...")
            perf_results = self._generate_performance_plot_micromamba(env_name, wheel_path, 
                                                                    python_version, actual_python)
            
            # Clean up environment
            self._run_subprocess(['micromamba', 'env', 'remove', '-n', env_name, '--yes'], 
                               capture_output=True)
            
            return {
                "wheel": wheel_path.name,
                "target_python": python_version,
                "actual_python": actual_python,
                "platform": self.platform_name,
                "architecture": platform.machine(),
                "status": "SUCCESS",
                "environment_type": "micromamba",
                **perf_results
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Test failed for {wheel_path.name}: {e}")
            logger.info(f"Keeping environment '{env_name}' for debugging. Remove manually with: micromamba env remove -n {env_name} --yes")
            
            return {
                "wheel": wheel_path.name,
                "target_python": python_version,
                "actual_python": "N/A",
                "platform": self.platform_name,
                "architecture": platform.machine(),
                "status": "FAILED",
                "environment_type": "micromamba",
                "error": str(e)
            }
    
    def _test_wheel_venv(self, wheel_path: Path, python_version: str) -> Dict:
        """Fallback: Test a wheel using virtual environment."""
        logger.info(f"🔧 Testing wheel using venv: {wheel_path.name}")
        
        # Create virtual environment
        venv_dir = self._create_venv(wheel_path, python_version)
        python_exe = self._get_venv_python(venv_dir)
        
        try:
            # Get actual Python version in venv
            logger.info("Verifying Python version in virtual environment...")
            result = self._run_subprocess(self._make_python_cmd(python_exe, ["--version"]), 
                                        capture_output=True, text=True, check=True)
            actual_python = result.stdout.strip().split()[1]
            logger.info(f"Virtual environment Python version: {actual_python}")
            
            # Install dependencies
            logger.info("Installing dependencies...")
            logger.info("  Upgrading pip...")
            self._run_subprocess(self._make_python_cmd(python_exe, ["-m", "pip", "install", "--upgrade", "pip"]), 
                               check=True, capture_output=True)
            
            logger.info("  Installing test dependencies...")
            self._run_subprocess(self._make_python_cmd(python_exe, ["-m", "pip", "install", "pytest", "numpy", "scipy", 
                                "matplotlib", "pyproj"]), check=True, capture_output=True)
            
            # Install the wheel
            logger.info(f"Installing wheel: {wheel_path.name}")
            logger.info(f"  Wheel path: {wheel_path}")
            logger.info(f"  Python executable: {python_exe}")
            
            try:
                install_cmd = self._make_python_cmd(python_exe, ["-m", "pip", "install", str(wheel_path.resolve())])
                logger.info(f"  Command: {' '.join(install_cmd)}")
                
                result = self._run_subprocess(install_cmd, check=True, capture_output=True, text=True)
                logger.info("  ✅ Wheel installed successfully")
                
                # Log pip install output for debugging
                if result.stdout.strip():
                    logger.debug(f"  pip install stdout: {result.stdout}")
                if result.stderr.strip():
                    logger.debug(f"  pip install stderr: {result.stderr}")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"  ❌ Wheel installation failed!")
                logger.error(f"  Return code: {e.returncode}")
                if e.stdout:
                    logger.error(f"  stdout: {e.stdout}")
                if e.stderr:
                    logger.error(f"  stderr: {e.stderr}")
                
                # Re-raise with more context
                raise subprocess.CalledProcessError(
                    e.returncode, 
                    e.cmd, 
                    f"Wheel installation failed: {e.stderr or e.stdout or 'No output'}"
                )
            
            # Run import tests
            logger.info("Running import tests...")
            self._run_import_tests(python_exe)
            
            # Run test suite
            logger.info("Running test suite...")
            self._run_test_suite(python_exe)
            
            # Generate performance visualization
            logger.info("Generating performance visualization...")
            perf_results = self._generate_performance_plot(python_exe, wheel_path, 
                                                         python_version, actual_python)
            
            # Clean up venv
            logger.info("Cleaning up virtual environment...")
            try:
                shutil.rmtree(venv_dir)
                logger.info(f"  ✅ Virtual environment removed: {venv_dir}")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not remove virtual environment: {e}")
            
            logger.info(f"✅ Wheel testing completed successfully: {wheel_path.name}")
            
            return {
                "wheel": wheel_path.name,
                "target_python": python_version,
                "actual_python": actual_python,
                "platform": self.platform_name,
                "architecture": platform.machine(),
                "status": "SUCCESS",
                "environment_type": "venv",
                **perf_results
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Test failed for {wheel_path.name}: {e}")
            logger.info(f"🔍 Keeping virtual environment for debugging: {venv_dir}")
            logger.info(f"   To investigate, run: {self._get_venv_python(venv_dir)} --version")
            logger.info(f"   To clean up manually, run: rm -rf {venv_dir}")
            
            return {
                "wheel": wheel_path.name,
                "target_python": python_version,
                "actual_python": "N/A",
                "platform": self.platform_name,
                "architecture": platform.machine(),
                "status": "FAILED",
                "environment_type": "venv",
                "error": str(e)
            }
    
    def _run_import_tests_micromamba(self, env_name: str):
        """Run import tests using micromamba environment."""
        script_path = Path(__file__).parent / "wheel_testing" / "import_test.py"
        
        result = self._run_in_micromamba_env(env_name, ["python", str(script_path)], 
                                           capture_output=True, text=True, check=True)
        
        logger.info("Import tests passed successfully")
        return result

    def _run_import_tests(self, python_exe: str):
        """Run import tests using specified Python executable."""
        script_path = Path(__file__).parent / "wheel_testing" / "import_test.py"
        
        result = self._run_subprocess(self._make_python_cmd(python_exe, [str(script_path)]), 
                                     capture_output=True, text=True, check=True)
        
        logger.info("Import tests passed successfully")
        return result

    def _run_test_suite_micromamba(self, env_name: str):
        """Run full test suite using micromamba environment."""
        script_path = Path(__file__).parent / "wheel_testing" / "test_suite.py"
        tests_dir = Path(__file__).parent.parent / "tests"
        
        result = self._run_in_micromamba_env(env_name, [
            "python", str(script_path), "--tests-dir", str(tests_dir)
        ], capture_output=True, text=True, check=True)
        
        logger.info("Test suite passed successfully")
        return result

    def _run_test_suite(self, python_exe: str):
        """Run full test suite using specified Python executable."""
        script_path = Path(__file__).parent / "wheel_testing" / "test_suite.py"
        tests_dir = Path(__file__).parent.parent / "tests"
        
        result = self._run_subprocess(self._make_python_cmd(python_exe, [str(script_path), "--tests-dir", str(tests_dir)]), 
                                     capture_output=True, text=True, check=True)
        
        logger.info("Test suite passed successfully")
        return result

    def _generate_performance_plot_micromamba(self, env_name: str, wheel_path: Path, 
                                            target_python: str, actual_python: str) -> Dict:
        """Generate performance visualization using micromamba environment."""
        script_path = Path(__file__).parent / "wheel_testing" / "performance_test.py"
        
        result = self._run_in_micromamba_env(env_name, [
            "python", str(script_path),
            "--wheel-name", wheel_path.name,
            "--target-python", target_python,
            "--platform-info", self.platform_name,
            "--output-dir", str(self.results_dir)
        ], capture_output=True, text=True, check=True)
        
        return self._parse_performance_results(result.stdout)

    def _generate_performance_plot(self, python_exe: str, wheel_path: Path, 
                                 target_python: str, actual_python: str) -> Dict:
        """Generate performance visualization (fallback)."""
        script_path = Path(__file__).parent / "wheel_testing" / "performance_test.py"
        
        result = self._run_subprocess(self._make_python_cmd(python_exe, [
            str(script_path),
            "--wheel-name", wheel_path.name,
            "--target-python", target_python,
            "--platform-info", self.platform_name,
            "--output-dir", str(self.results_dir)
        ]), capture_output=True, text=True, check=True)
        
        return self._parse_performance_results(result.stdout)
    
    def _parse_performance_results(self, output: str) -> Dict:
        """Parse performance results from script output."""
        for line in output.split('\n'):
            if line.startswith('PERF_RESULTS:'):
                parts = line.split(':')
                exec_time = float(parts[1])
                grid_size = int(parts[2])
                num_triangles = int(parts[3])
                
                return {
                    "execution_time": exec_time,
                    "grid_size": grid_size,
                    "num_triangles": num_triangles
                }
        
        raise RuntimeError("Could not parse performance results")
    
    def save_results(self, results: List[Dict], filename: str = "test_results.json"):
        """Save test results to JSON file."""
        results_file = self.results_dir / filename
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    
    def test_all_wheels(self) -> List[Dict]:
        """Discover and test all compatible wheels."""
        wheels = self.discover_wheels()
        if not wheels:
            logger.warning("No compatible wheels found")
            return []
        
        results = []
        for wheel_path, python_version in wheels:
            try:
                result = self.test_wheel(wheel_path, python_version)
                results.append(result)
                env_type = result.get('environment_type', 'unknown')
                status = result['status']
                emoji = "✅" if status == "SUCCESS" else "❌"
                logger.info(f"{emoji} {wheel_path.name} - {status} ({env_type})")
            except Exception as e:
                logger.error(f"❌ {wheel_path.name} - {e}")
                results.append({
                    "wheel": wheel_path.name,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test cutde wheels with micromamba")
    parser.add_argument("--wheels-dir", type=Path, required=True,
                       help="Directory containing wheels to test")
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Directory to save test results")
    parser.add_argument("--platform", type=str,
                       help="Platform name (auto-detected if not provided)")
    parser.add_argument("--single-wheel", type=Path,
                       help="Test only a specific wheel file")
    parser.add_argument("--python-version", type=str,
                       help="Python version to use (extracted from wheel if not provided)")
    parser.add_argument("--force-micromamba", action="store_true",
                       help="Force use of micromamba instead of venv")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging (debug level)")
    
    args = parser.parse_args()
    
    # Configure logging based on verbose argument
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        # Add more detailed format for debug logging
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)
        logger.debug("Verbose logging enabled.")
    
    tester = WheelTester(args.wheels_dir, args.results_dir, args.platform)
    
    # Override to force micromamba if requested
    if args.force_micromamba:
        if not tester.micromamba_available:
            logger.error("Cannot force micromamba: micromamba is not available")
            return 1
        tester.force_micromamba = True
        logger.info("Forcing use of micromamba instead of venv")
    
    if args.single_wheel:
        python_version = args.python_version
        if not python_version:
            # Extract from wheel filename
            match = re.search(r"cp(\d+)", args.single_wheel.name)
            if match:
                python_ver = match.group(1)
                python_version = f"{python_ver[0]}.{python_ver[1:]}"
            else:
                python_version = "3.11"  # default
        
        # Resolve the wheel path correctly
        single_wheel_path = args.single_wheel
        
        # If the path doesn't exist, try relative to wheels_dir
        if not single_wheel_path.exists():
            potential_path = args.wheels_dir / single_wheel_path.name
            if potential_path.exists():
                single_wheel_path = potential_path
                logger.info(f"Found wheel in wheels directory: {single_wheel_path}")
            else:
                logger.error(f"Wheel file not found: {args.single_wheel}")
                logger.error(f"Tried: {args.single_wheel} and {potential_path}")
                return 1
        else:
            logger.info(f"Using wheel at specified path: {single_wheel_path}")
        
        results = [tester.test_wheel(single_wheel_path, python_version)]
    else:
        results = tester.test_all_wheels()
    
    tester.save_results(results)
    
    # Print summary
    successful = sum(1 for r in results if r.get("status") == "SUCCESS")
    total = len(results)
    env_types = [r.get("environment_type", "unknown") for r in results if r.get("environment_type")]
    env_summary = ", ".join(set(env_types)) if env_types else "unknown"
    
    logger.info(f"Testing complete: {successful}/{total} wheels successful")
    logger.info(f"Environment types used: {env_summary}")
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main()) 