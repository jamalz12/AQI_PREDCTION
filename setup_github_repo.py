#!/usr/bin/env python3
"""
GitHub Repository Setup Script for Karachi AQI Project
This script helps you initialize Git, create a GitHub repository, and push your code.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and return success status."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr

def main():
    print("🚀 Setting up GitHub repository for Karachi AQI Project")
    print("=" * 60)

    # Check if git is initialized
    if not Path('.git').exists():
        print("📝 Initializing Git repository...")
        success, _ = run_command("git init", "Initializing Git repository")
        if not success:
            print("❌ Failed to initialize Git. Please install Git and try again.")
            return

        success, _ = run_command("git add .", "Adding all files to Git")
        if not success:
            return

        success, _ = run_command('git commit -m "Initial commit: Karachi AQI prediction system"', "Creating initial commit")
        if not success:
            return
    else:
        print("✅ Git repository already initialized")

    # Check GitHub CLI or provide manual instructions
    print("\n📋 Next steps for GitHub setup:")
    print("1. Create a new repository on GitHub:")
    print("   - Go to https://github.com/new")
    print("   - Repository name: karachi-aqi-prediction")
    print("   - Make it public or private (your choice)")
    print("   - Don't initialize with README, .gitignore, or license")

    repo_url = input("\n2. Enter your GitHub repository URL (e.g., https://github.com/yourusername/karachi-aqi-prediction.git): ").strip()

    if not repo_url:
        print("❌ Repository URL is required. Please create a GitHub repository first.")
        return

    # Add remote origin
    success, _ = run_command(f"git remote add origin {repo_url}", "Adding GitHub remote")
    if not success:
        # If remote already exists, update it
        success, _ = run_command(f"git remote set-url origin {repo_url}", "Updating GitHub remote")

    # Push to GitHub
    success, _ = run_command("git push -u origin main", "Pushing code to GitHub")
    if not success:
        # Try with master branch
        success, _ = run_command("git push -u origin master", "Pushing code to GitHub (master branch)")
        if not success:
            print("❌ Failed to push to GitHub. Please check your repository URL and permissions.")
            return

    print("\n🎉 Repository setup complete!")
    print("\n📝 Next steps:")
    print("1. Go to your GitHub repository")
    print("2. Click on 'Settings' tab")
    print("3. Click on 'Secrets and variables' → 'Actions'")
    print("4. Click 'New repository secret'")
    print("5. Name: OPENWEATHER_API_KEY")
    print("6. Value: da06b92d3139ce209b04dba2132ad4ce")
    print("7. Click 'Add secret'")
    print("\n🚀 Your CI/CD pipeline will now run automatically:")
    print("   - Data collection: Every hour")
    print("   - Model training: Daily at 2 AM UTC")
    print("   - You can also trigger manually from the Actions tab")

    print("\n📊 To monitor your pipeline:")
    print("   - Go to your GitHub repository")
    print("   - Click on 'Actions' tab")
    print("   - You'll see the workflow runs and their status")

if __name__ == "__main__":
    main()
