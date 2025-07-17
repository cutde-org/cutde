#!/usr/bin/env python3
"""
Test script to verify that all dependencies and tools are properly set up.
Run this before using the wheel testing scripts.
"""

import subprocess
import sys
import shutil
from pathlib import Path


def check_command(cmd, name, required=True):
    """Check if a command is available."""
    try:
        result = subprocess.run([cmd, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print(f"✅ {name}: {version}")
            return True
        else:
            print(f"❌ {name}: Command failed")
            return False
    except FileNotFoundError:
        if required:
            print(f"❌ {name}: Not found (required)")
        else:
            print(f"⚠️  {name}: Not found (optional)")
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ {name}: Command timed out")
        return False
    except Exception as e:
        print(f"❌ {name}: Error - {e}")
        return False


def check_python_modules():
    """Check if required Python modules are available."""
    modules = {
        'matplotlib': 'required for plotting',
        'numpy': 'required for computations',
        'json': 'built-in module',
        'pathlib': 'built-in module'
    }
    
    print("\nChecking Python modules:")
    all_good = True
    
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"✅ {module}: Available ({description})")
        except ImportError:
            print(f"❌ {module}: Not available ({description})")
            all_good = False
    
    return all_good


def check_gh_auth():
    """Check GitHub CLI authentication."""
    try:
        result = subprocess.run(['gh', 'auth', 'status'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ GitHub CLI: Authenticated")
            return True
        else:
            print("⚠️  GitHub CLI: Not authenticated (run 'gh auth login')")
            return False
    except FileNotFoundError:
        print("❌ GitHub CLI: Not found")
        return False
    except Exception as e:
        print(f"❌ GitHub CLI auth: Error - {e}")
        return False


def main():
    print("🔍 Checking wheel testing setup...\n")
    
    print("Checking required tools:")
    python_ok = check_command('python', 'Python', required=True)
    gh_ok = check_command('gh', 'GitHub CLI', required=True)
    
    print("\nChecking optional tools:")
    micromamba_ok = check_command('micromamba', 'micromamba', required=False)
    
    # Check Python modules
    modules_ok = check_python_modules()
    
    # Check GitHub authentication
    print("\nChecking authentication:")
    auth_ok = check_gh_auth()
    
    # Check if scripts exist
    print("\nChecking scripts:")
    scripts_dir = Path(__file__).parent
    script_files = ['test_wheels.py', 'download_artifacts.py', 'generate_summary.py']
    scripts_ok = True
    
    for script in script_files:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✅ {script}: Found")
        else:
            print(f"❌ {script}: Not found")
            scripts_ok = False
    
    # Summary
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    essential_ok = python_ok and gh_ok and modules_ok and scripts_ok
    
    if essential_ok:
        print("✅ Essential components: All good!")
    else:
        print("❌ Essential components: Issues found")
    
    if micromamba_ok:
        print("✅ Environment management: micromamba available (recommended)")
    else:
        print("⚠️  Environment management: Will use venv fallback")
    
    if auth_ok:
        print("✅ Authentication: GitHub CLI authenticated")
    else:
        print("⚠️  Authentication: Run 'gh auth login' for private repos")
    
    print("\nNext steps:")
    if not essential_ok:
        print("1. Install missing required components")
        if not gh_ok:
            print("   - Install gh CLI: https://cli.github.com/")
        if not modules_ok:
            print("   - Install Python modules: pip install -r requirements.txt")
    
    if not auth_ok:
        print("2. Authenticate GitHub CLI: gh auth login")
    
    if not micromamba_ok:
        print("3. (Optional) Install micromamba for better Python management:")
        print("   https://mamba.readthedocs.io/en/latest/installation.html")
    
    if essential_ok:
        print("\n🎉 Ready to test wheels!")
        print("\nExample usage:")
        print("  python scripts/download_artifacts.py <run-url> --wheels-only")
        print("  python scripts/test_wheels.py --wheels-dir ./wheels --results-dir ./results")
        print("  python scripts/generate_summary.py --results-dir ./results --format console")
    
    return 0 if essential_ok else 1


if __name__ == "__main__":
    sys.exit(main()) 