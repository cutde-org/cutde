# TDE Wheel Testing Scripts

## ⚠️ **DISCLAIMER: VIBE-CODED SOFTWARE** ⚠️

**WARNING**: Everything in this directory was shamelessly vibe-coded! **Don't overthink it!**

We just need to ensure that the wheels are not broken, so we can afford to be somewhat careless.

---

This directory contains Python scripts for comprehensive testing of cutde wheels across platforms and Python versions, using modern tools for better reliability and ease of use.

## 🚀 Key Features

- **gh CLI integration**: Simplified authentication and artifact downloading
- **micromamba environments**: Reliable Python version management
- **Graceful fallbacks**: Works with standard Python venv if micromamba unavailable
- **Cross-platform support**: Linux, macOS, Windows with all architectures
- **Performance benchmarking**: Visual plots with timing data for each wheel

## Prerequisites

### Required
- Python 3.8+ (for running the scripts)
- [gh CLI](https://cli.github.com/) (for artifact downloading)

### Recommended
- [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) (for reliable Python environment management)

### Installation Commands
```bash
# Install gh CLI (choose your platform)
# macOS
brew install gh

# Ubuntu/Debian
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh

# Windows (with winget)
winget install --id GitHub.cli

# Install micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
# Or follow: https://mamba.readthedocs.io/en/latest/installation.html

# Authenticate with GitHub
gh auth login
```

## Scripts Overview

### `test_wheels.py`
Main script for discovering, testing, and benchmarking wheels with micromamba.

**Features:**
- **Automatic platform detection** with support for manylinux/musllinux variants
- **micromamba integration** for precise Python version management
- **Graceful fallback** to standard venv if micromamba unavailable
- **Virtual environment isolation** for each test
- **Performance benchmarking** with visualization
- **Comprehensive error handling** and logging

**Usage:**
```bash
# Test all wheels in a directory (with micromamba)
python test_wheels.py --wheels-dir ./wheels --results-dir ./results

# Test a specific wheel
python test_wheels.py --wheels-dir ./wheels --results-dir ./results --single-wheel cutde-0.4.3-cp311-cp311-linux_x86_64.whl

# Force use of micromamba instead of venv
python test_wheels.py --wheels-dir ./wheels --results-dir ./results --force-micromamba

# Enable verbose debugging (helpful for troubleshooting)
python test_wheels.py --wheels-dir ./wheels --results-dir ./results --verbose

# Override platform detection
python test_wheels.py --wheels-dir ./wheels --results-dir ./results --platform linux-x86_64
```

### `download_artifacts.py`
Download GitHub Actions artifacts using gh CLI (much simpler than API calls).

**Features:**
- **gh CLI integration** - uses your existing GitHub authentication
- **Automatic extraction** - no manual zip handling
- **Wheel organization** - flattens directory structure automatically
- **Pattern filtering** - target specific artifact types
- **Multiple download modes** - individual artifacts or bulk download

**Usage:**
```bash
# Download all artifacts (fastest)
python download_artifacts.py https://github.com/user/repo/actions/runs/123456789 --all

# Download only wheel artifacts
python download_artifacts.py https://github.com/user/repo/actions/runs/123456789 --wheels-only

# List available artifacts
python download_artifacts.py https://github.com/user/repo/actions/runs/123456789 --list

# Filter by pattern
python download_artifacts.py https://github.com/user/repo/actions/runs/123456789 --pattern "cibw-wheels"

# Note: Uses gh auth login credentials automatically
```

### `generate_summary.py`
Generate performance summaries from test results (unchanged).

**Features:**
- **Multiple output formats** (console, markdown, GitHub Actions)
- **Performance statistics** with environment type tracking
- **Platform-specific breakdowns**
- **Visual plot listings**

**Usage:**
```bash
# Generate console summary
python generate_summary.py --results-dir ./results --format console

# Generate markdown report
python generate_summary.py --results-dir ./results --output summary.md

# Write to GitHub Actions summary
python generate_summary.py --results-dir ./results --github-summary
```

## Complete Local Testing Workflow

1. **Authenticate with GitHub:**
   ```bash
   gh auth login
   ```

2. **Download wheels from a GitHub Actions run:**
   ```bash
   python cibuildwheel-scripts/download_artifacts.py https://github.com/user/repo/actions/runs/123456789 --wheels-only -o ./local-test
   ```

3. **Test all wheels:**
   ```bash
   python cibuildwheel-scripts/test_wheels.py --wheels-dir ./local-test/wheels --results-dir ./local-test/results
   ```

4. **Generate summary:**
   ```bash
   python cibuildwheel-scripts/generate_summary.py --results-dir ./local-test/results --format console
   ```

## Installation

```bash
# Install script dependencies
pip install -r cibuildwheel-scripts/requirements.txt

# Make scripts executable (Unix-like systems)
chmod +x cibuildwheel-scripts/*.py
```

## Environment Management Comparison

### With micromamba (Recommended)
- ✅ **Precise Python versions**: Can install exact Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
- ✅ **Fast environment creation**: Parallel package resolution
- ✅ **Reliable dependencies**: conda-forge packages well-tested
- ✅ **Cross-platform consistency**: Same behavior across OS
- ✅ **Easy cleanup**: `micromamba env remove` is instant

### With venv (Fallback)
- ⚠️ **Limited Python versions**: Only what's already installed on system
- ⚠️ **Dependency conflicts**: pip resolution can be inconsistent
- ✅ **No additional tools**: Works with standard Python installation
- ✅ **Lightweight**: No extra dependencies

## Environment Variables

- `GITHUB_STEP_SUMMARY`: Path to GitHub Actions summary file (set automatically in CI)

## Output Files

### Test Results
- `test_results.json`: Structured test results with environment type tracking
- `tde_test_*.png`: Performance visualization plots
- Environment logs (temporary, cleaned up automatically)

### Summary Formats
- **Console**: Human-readable terminal output
- **Markdown**: Rich formatted report with tables
- **GitHub**: Formatted for GitHub Actions summary

## Platform Support

The scripts automatically detect and support:
- **Linux**: x86_64, aarch64 (manylinux, musllinux variants)
- **macOS**: Intel (x86_64), Apple Silicon (arm64)
- **Windows**: AMD64

## Python Version Support

- **With micromamba**: Automatically creates environments for Python 3.8-3.13 based on wheel filename tags
- **With venv**: Uses available Python versions on the system

## Error Handling

- **micromamba fallbacks**: If exact Python version unavailable, tries major version (e.g., 3.11 → 3.x)
- **venv fallbacks**: Uses best available Python if target version missing
- **Graceful failures**: Continues testing other wheels if one fails
- **Environment cleanup**: Automatic cleanup on both success and failure
- **Comprehensive logging**: Detailed output for debugging
- **Structured results**: JSON format with environment type tracking

## CI/CD Integration

The scripts work seamlessly in GitHub Actions:

```yaml
- name: Install script dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r cibuildwheel-scripts/requirements.txt

- name: Test all wheels
  run: |
    python cibuildwheel-scripts/test_wheels.py --wheels-dir ./wheels-flat --results-dir ./wheel-results

- name: Generate summary
  run: |
    python cibuildwheel-scripts/generate_summary.py --results-dir ./wheel-results --github-summary
```

**Benefits in CI:**
1. **No authentication setup**: gh CLI uses GitHub Actions token automatically
2. **Fast environment creation**: micromamba cached in runners
3. **Reliable Python versions**: Consistent across all runners
4. **Rich summaries**: Detailed performance comparison tables
5. **Artifact management**: All plots and results preserved

## Troubleshooting

### gh CLI Issues
```bash
# Check authentication
gh auth status

# Re-authenticate if needed
gh auth login

# Test with a public repo first
python cibuildwheel-scripts/download_artifacts.py https://github.com/public/repo/actions/runs/123 --list
```

### Wheel Testing Issues

If wheel testing fails, use the `--verbose` flag to see detailed debugging output:

```bash
python test_wheels.py --wheels-dir ./wheels --results-dir ./results --verbose
```

The verbose output will show:
- **Python executable discovery**: Which Python executables are found and tested
- **Version compatibility checks**: Exact version comparisons and why executables are accepted/rejected
- **Environment selection rationale**: Why venv vs micromamba was chosen
- **Command execution details**: All commands run with their output
- **Wheel installation process**: Detailed pip install logs with error messages

Common issues and solutions:
- **"No compatible Python X.Y found"**: The system doesn't have the required Python version. Install it or use `--force-micromamba` to let micromamba create the environment.
- **"Wheel installation failed"**: Check the detailed pip error output in verbose mode. Often due to missing system dependencies.
- **"Version mismatch"**: The script found a Python executable but it's the wrong version. This usually means the executable search logic needs adjustment.

### micromamba Issues
```bash
# Test micromamba installation
micromamba --version

# Initialize if needed
micromamba shell init

# Force micromamba usage
python cibuildwheel-scripts/test_wheels.py --wheels-dir ./wheels --results-dir ./results --force-micromamba
```

### Permission Issues
```bash
# Make scripts executable
chmod +x cibuildwheel-scripts/*.py

# Or run with python explicitly
python cibuildwheel-scripts/test_wheels.py --help
```
