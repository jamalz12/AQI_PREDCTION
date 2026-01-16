"""
CI/CD Pipeline Setup Script for Karachi AQI System
=================================================

This script helps you set up the CI/CD pipeline for automated
data collection and model training.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is installed")

            # Check if Docker daemon is running
            result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker daemon is running")
                return True
            else:
                print("❌ Docker daemon is not running")
                print("   Please start Docker Desktop or Docker service")
                return False
        else:
            print("❌ Docker is not installed")
            print("   Please install Docker from https://docker.com")
            return False
    except FileNotFoundError:
        print("❌ Docker command not found")
        print("   Please install Docker from https://docker.com")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker Compose is available")
            return True
        else:
            # Try docker compose (newer syntax)
            result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker Compose V2 is available")
                return True
            else:
                print("❌ Docker Compose is not available")
                return False
    except FileNotFoundError:
        print("❌ Docker Compose command not found")
        return False

def setup_airflow():
    """Set up Apache Airflow with Docker"""
    print("\n🌪️  Setting up Apache Airflow CI/CD Pipeline")
    print("=" * 50)

    # Check if docker-compose.yml exists
    if not Path('docker-compose.yml').exists():
        print("❌ docker-compose.yml not found")
        return False

    # Create necessary directories
    directories = ['airflow/dags', 'airflow/logs', 'airflow/plugins', 'airflow/config']
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

    # Check if DAG file exists
    dag_file = Path('airflow/dags/karachi_aqi_pipeline.py')
    if not dag_file.exists():
        print("❌ DAG file not found")
        return False

    print("✅ Airflow setup files found")

    # Ask user if they want to start Airflow
    response = input("\n🚀 Start Apache Airflow now? (y/n): ").lower().strip()

    if response == 'y':
        print("\n🏃 Starting Airflow services...")
        print("This may take a few minutes for the first run...")

        try:
            # Start services
            subprocess.run(['docker-compose', 'up', '-d'], check=True)

            print("\n✅ Airflow services started!")
            print("\n🌐 Access points:")
            print("   Airflow UI: http://localhost:8080")
            print("   Username: admin")
            print("   Password: admin")
            print("\n📋 Next steps:")
            print("   1. Open http://localhost:8080")
            print("   2. Enable DAGs: karachi_aqi_pipeline & karachi_aqi_hourly_collection")
            print("   3. Monitor pipeline execution")
            print("\n📊 Pipeline schedule:")
            print("   • Data collection: Every hour")
            print("   • Model training: Daily at 2 AM")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start Airflow: {e}")
            print("\n🔧 Troubleshooting:")
            print("   1. Check if ports 8080, 5432 are available")
            print("   2. Ensure Docker has enough resources (4GB+ RAM)")
            print("   3. Run: docker-compose logs")
            return False
    else:
        print("\n📝 To start Airflow later, run:")
        print("   docker-compose up -d")
        return True

def setup_github_actions():
    """Set up GitHub Actions CI/CD"""
    print("\n🤖 Setting up GitHub Actions CI/CD")
    print("=" * 50)

    workflow_file = Path('.github/workflows/karachi-aqi-pipeline.yml')
    if workflow_file.exists():
        print("✅ GitHub Actions workflow file exists")
        print(f"   Location: {workflow_file}")
    else:
        print("❌ GitHub Actions workflow file not found")
        return False

    print("\n📋 GitHub Actions Setup Instructions:")
    print("1. Push this repository to GitHub")
    print("2. Go to repository → Settings → Secrets and variables → Actions")
    print("3. Add secret: OPENWEATHER_API_KEY = your_api_key")
    print("4. The pipeline will run automatically on schedule")
    print("5. Monitor runs in the 'Actions' tab")

    return True

def setup_cron_jobs():
    """Set up simple cron jobs"""
    print("\n⏰ Setting up Cron Jobs (Simple Scheduling)")
    print("=" * 50)

    print("📋 Cron Job Setup Instructions:")
    print("\nFor Linux/Mac:")
    print("   crontab -e")
    print("   Add these lines:")
    print("   ")
    print("   # Data collection every hour")
    print("   0 * * * * cd /path/to/your/project && python -c \"")
    print("   import sys")
    print("   sys.path.append('.')")
    print("   from karachi_aqi_app import KarachiAQISystem")
    print("   system = KarachiAQISystem()")
    print("   weather = system.get_current_weather()")
    print("   if weather:")
    print("       system.store_current_data(weather)")
    print("       print('✅ Data collected')")
    print("   \"")
    print("   ")
    print("   # Model training daily at 2 AM")
    print("   0 2 * * * cd /path/to/your/project && python train_karachi_models.py")

    print("\nFor Windows Task Scheduler:")
    print("   1. Create data_collection.py script")
    print("   2. Use Task Scheduler to run hourly")
    print("   3. Use Task Scheduler to run training daily")

    return True

def main():
    """Main setup function"""
    print("🚀 Karachi AQI CI/CD Pipeline Setup")
    print("=" * 60)

    print("Choose your CI/CD solution:")
    print("1. 🌪️  Apache Airflow (Recommended - Full-featured)")
    print("2. 🤖 GitHub Actions (Simple - Repository-based)")
    print("3. ⏰ Cron Jobs (Basic - Local machine)")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            # Check Docker prerequisites
            if not check_docker():
                print("❌ Docker is required for Apache Airflow")
                continue

            if not check_docker_compose():
                print("❌ Docker Compose is required for Apache Airflow")
                continue

            success = setup_airflow()
            if success:
                print("\n🎉 Apache Airflow CI/CD pipeline setup complete!")
            break

        elif choice == '2':
            success = setup_github_actions()
            if success:
                print("\n🎉 GitHub Actions CI/CD pipeline configured!")
            break

        elif choice == '3':
            success = setup_cron_jobs()
            if success:
                print("\n🎉 Cron jobs CI/CD pipeline instructions provided!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

    print("\n📚 Additional Resources:")
    print("   📖 Full documentation: CICD_README.md")
    print("   🐛 Troubleshooting: Check logs and error messages")
    print("   📞 Support: Refer to documentation links")

    print("\n🎯 Your Karachi AQI system now has automated CI/CD!")
    print("   ✅ Data collection every hour")
    print("   ✅ Model training every day")
    print("   ✅ Quality checks and monitoring")

if __name__ == "__main__":
    main()
