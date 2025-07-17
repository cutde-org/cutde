#!/usr/bin/env python3
"""
Download GitHub Actions artifacts from a run URL using gh CLI.
Much simpler authentication and artifact handling.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GHArtifactDownloader:
    """Download artifacts from GitHub Actions runs using gh CLI."""
    
    def __init__(self):
        self._check_gh_cli()
    
    def _check_gh_cli(self):
        """Check if gh CLI is available and authenticated."""
        try:
            result = subprocess.run(['gh', '--version'], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"Using {result.stdout.strip().split()[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("gh CLI not found. Please install: https://cli.github.com/")
        
        # Check authentication
        try:
            result = subprocess.run(['gh', 'auth', 'status'], 
                                  capture_output=True, text=True, check=True)
            logger.info("gh CLI is authenticated")
        except subprocess.CalledProcessError:
            logger.warning("gh CLI not authenticated. Run 'gh auth login' first.")
            # Continue anyway - might work for public repos
    
    def parse_run_url(self, url: str) -> Dict[str, str]:
        """Parse a GitHub Actions run URL to extract repo and run ID."""
        import re
        # Example: https://github.com/owner/repo/actions/runs/123456789
        pattern = r'https://github\.com/([^/]+)/([^/]+)/actions/runs/(\d+)'
        match = re.match(pattern, url)
        
        if not match:
            raise ValueError(f"Invalid GitHub Actions run URL: {url}")
        
        owner, repo, run_id = match.groups()
        return {
            'owner': owner,
            'repo': repo,
            'run_id': run_id,
            'repo_full': f'{owner}/{repo}'
        }
    
    def list_artifacts(self, run_url: str) -> List[Dict]:
        """List all artifacts for a GitHub Actions run."""
        run_info = self.parse_run_url(run_url)
        
        logger.info(f"Listing artifacts for run {run_info['run_id']} in {run_info['repo_full']}")
        
        try:
            # Use gh api to get artifacts since gh run view doesn't support --json artifacts
            api_url = f"/repos/{run_info['repo_full']}/actions/runs/{run_info['run_id']}/artifacts"
            result = subprocess.run([
                'gh', 'api', api_url
            ], capture_output=True, text=True, check=True)
            
            data = json.loads(result.stdout)
            artifacts = data.get('artifacts', [])
            
            logger.info(f"Found {len(artifacts)} artifacts")
            return artifacts
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list artifacts: {e}")
            logger.error(f"gh CLI output: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse artifacts JSON: {e}")
            raise
    
    def download_artifact(self, run_url: str, artifact_name: str, output_dir: Path) -> Path:
        """Download a specific artifact."""
        run_info = self.parse_run_url(run_url)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading artifact: {artifact_name}")
        
        try:
            # gh CLI downloads and extracts automatically
            result = subprocess.run([
                'gh', 'run', 'download', run_info['run_id'],
                '--repo', run_info['repo_full'],
                '--name', artifact_name,
                '--dir', str(output_dir)
            ], capture_output=True, text=True, check=True)
            
            # The artifact is extracted to output_dir/artifact_name/
            artifact_dir = output_dir / artifact_name
            
            if artifact_dir.exists():
                logger.info(f"Downloaded and extracted to: {artifact_dir}")
                return artifact_dir
            else:
                # Sometimes gh extracts directly to output_dir
                logger.info(f"Downloaded to: {output_dir}")
                return output_dir
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download artifact {artifact_name}: {e}")
            logger.error(f"gh CLI output: {e.stderr}")
            raise
    
    def download_all_artifacts(self, run_url: str, output_dir: Path, 
                             pattern: Optional[str] = None) -> List[Path]:
        """Download all artifacts from a run, optionally filtered by pattern."""
        import re
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = self.list_artifacts(run_url)
        
        if pattern:
            artifacts = [a for a in artifacts if re.search(pattern, a['name'])]
            logger.info(f"Filtered to {len(artifacts)} artifacts matching pattern: {pattern}")
        
        if not artifacts:
            logger.warning("No artifacts found to download")
            return []
        
        logger.info(f"Downloading {len(artifacts)} artifacts...")
        
        downloaded_paths = []
        for artifact in artifacts:
            try:
                path = self.download_artifact(run_url, artifact['name'], output_dir)
                downloaded_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to download artifact {artifact['name']}: {e}")
        
        return downloaded_paths
    
    def download_wheels(self, run_url: str, output_dir: Path) -> Path:
        """Download wheel artifacts and organize them in a flat structure."""
        wheels_dir = Path(output_dir) / "wheels"
        wheels_dir.mkdir(parents=True, exist_ok=True)
        
        # Download wheel artifacts
        artifact_dirs = self.download_all_artifacts(run_url, output_dir, pattern=r'cibw-wheels-')
        
        # Flatten wheel structure
        wheel_count = 0
        for artifact_dir in artifact_dirs:
            for wheel_file in artifact_dir.rglob("*.whl"):
                dest_path = wheels_dir / wheel_file.name
                if dest_path.exists():
                    logger.warning(f"Wheel already exists, skipping: {wheel_file.name}")
                    continue
                
                shutil.copy2(wheel_file, dest_path)
                wheel_count += 1
                logger.info(f"Copied wheel: {wheel_file.name}")
        
        logger.info(f"Organized {wheel_count} wheels in {wheels_dir}")
        return wheels_dir
    
    def download_all_run_artifacts(self, run_url: str, output_dir: Path) -> Path:
        """Download ALL artifacts from a run using gh run download (simpler approach)."""
        run_info = self.parse_run_url(run_url)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading all artifacts from run {run_info['run_id']}")
        
        try:
            result = subprocess.run([
                'gh', 'run', 'download', run_info['run_id'],
                '--repo', run_info['repo_full'],
                '--dir', str(output_dir)
            ], capture_output=True, text=True, check=True)
            
            logger.info(f"All artifacts downloaded to: {output_dir}")
            return output_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download artifacts: {e}")
            logger.error(f"gh CLI output: {e.stderr}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Download GitHub Actions artifacts using gh CLI",
        epilog="Make sure to run 'gh auth login' first for private repositories."
    )
    parser.add_argument("run_url", help="GitHub Actions run URL")
    parser.add_argument("--output-dir", "-o", type=Path, default="./artifacts",
                       help="Directory to save artifacts")
    parser.add_argument("--pattern", "-p", help="Pattern to filter artifact names")
    parser.add_argument("--wheels-only", action="store_true",
                       help="Download and organize only wheel artifacts")
    parser.add_argument("--list", action="store_true",
                       help="List artifacts without downloading")
    parser.add_argument("--all", action="store_true",
                       help="Download all artifacts in one command (faster)")
    
    args = parser.parse_args()
    
    downloader = GHArtifactDownloader()
    
    try:
        if args.list:
            # List artifacts
            artifacts = downloader.list_artifacts(args.run_url)
            print(f"\nFound {len(artifacts)} artifacts:")
            for artifact in artifacts:
                size_mb = artifact.get('size_in_bytes', 0) / (1024 * 1024)
                expired = " (EXPIRED)" if artifact.get('expired', False) else ""
                print(f"  - {artifact['name']} ({size_mb:.1f} MB){expired}")
        
        elif args.wheels_only:
            # Download and organize wheels
            wheels_dir = downloader.download_wheels(args.run_url, args.output_dir)
            print(f"\nWheels organized in: {wheels_dir}")
            
            # List downloaded wheels
            wheels = list(wheels_dir.glob("*.whl"))
            print(f"Downloaded {len(wheels)} wheels:")
            for wheel in sorted(wheels):
                print(f"  - {wheel.name}")
        
        elif args.all:
            # Download all artifacts at once
            output_dir = downloader.download_all_run_artifacts(args.run_url, args.output_dir)
            print(f"\nAll artifacts downloaded to: {output_dir}")
            
            # Count artifacts
            artifact_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            print(f"Downloaded {len(artifact_dirs)} artifacts:")
            for artifact_dir in sorted(artifact_dirs):
                file_count = len(list(artifact_dir.rglob("*")))
                print(f"  - {artifact_dir.name} ({file_count} files)")
        
        else:
            # Download filtered artifacts
            paths = downloader.download_all_artifacts(args.run_url, args.output_dir, args.pattern)
            print(f"\nDownloaded {len(paths)} artifacts to: {args.output_dir}")
            for path in paths:
                print(f"  - {path.name}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 