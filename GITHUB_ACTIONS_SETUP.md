# GitHub Actions Setup Guide for Karachi AQI Project

This guide will help you set up GitHub Actions for automated CI/CD pipeline for your Karachi AQI prediction system.

## 🚀 Quick Setup

### 1. Run the Setup Script

```bash
python setup_github_repo.py
```

This script will:
- Initialize Git (if not already done)
- Help you create a GitHub repository
- Push your code to GitHub
- Provide instructions for adding secrets

### 2. Add GitHub Secrets

After pushing your code to GitHub:

1. Go to your GitHub repository
2. Click on **"Settings"** tab
3. Click on **"Secrets and variables"** → **"Actions"**
4. Click **"New repository secret"**
5. Add the following secret:
   - **Name:** `OPENWEATHER_API_KEY`
   - **Value:** `da06b92d3139ce209b04dba2132ad4ce`
6. Click **"Add secret"**

## 🔄 CI/CD Pipeline Overview

The GitHub Actions pipeline includes:

### **Automated Data Collection** (Every hour)
- Collects current weather data from OpenWeather API
- Stores data in MongoDB (with CSV fallback)
- Performs data quality checks

### **Automated Model Training** (Daily at 2 AM UTC)
- Checks if new training is needed (avoids unnecessary retraining)
- Trains Linear Regression, Random Forest, and XGBoost models
- Evaluates model performance
- Saves trained models as artifacts

### **Manual Triggers**
- You can manually trigger the pipeline from GitHub Actions tab
- Useful for testing or immediate updates

## 📊 Monitoring Your Pipeline

### View Pipeline Runs
1. Go to your GitHub repository
2. Click on **"Actions"** tab
3. You'll see all workflow runs with their status:
   - 🟢 **Success**: Pipeline completed successfully
   - 🔴 **Failure**: Something went wrong
   - 🟡 **In Progress**: Currently running

### Pipeline Jobs
Each pipeline run consists of:
1. **Data Collection Job**: Collects and stores weather data
2. **Model Training Job**: Trains and evaluates models
3. **Deploy Job**: Confirms successful completion

## 🔧 Pipeline Configuration

The pipeline is configured in `.github/workflows/karachi-aqi-pipeline.yml`:

```yaml
# Triggers
on:
  schedule:
    - cron: '0 * * * *'    # Data collection: Every hour
    - cron: '0 2 * * *'    # Model training: Daily at 2 AM UTC
  workflow_dispatch:       # Manual trigger
  push:                    # Trigger on code changes
```

## 📈 Pipeline Features

### Smart Training Logic
- Only retrains models if needed (no recent models exist)
- Saves computational resources and avoids unnecessary runs

### Data Quality Checks
- Validates data collection success
- Checks for missing values
- Ensures data freshness

### Model Artifacts
- Trained models are saved as GitHub artifacts
- Available for download from successful runs
- Retention period: 30 days

### Error Handling
- Comprehensive error logging
- Graceful handling of API failures
- Automatic retries for transient issues

## 🛠️ Troubleshooting

### Pipeline Not Running
1. Check if secrets are properly configured
2. Verify repository has GitHub Actions enabled (should be by default)
3. Check if cron schedules are correct

### Data Collection Failing
1. Verify OpenWeather API key is valid
2. Check API rate limits
3. Ensure network connectivity

### Model Training Failing
1. Check if data is available for training
2. Verify dependencies are installed
3. Check model training logs for specific errors

### Manual Trigger
To manually run the pipeline:
1. Go to **"Actions"** tab
2. Click **"Karachi AQI CI/CD Pipeline"**
3. Click **"Run workflow"**
4. Choose branch and click **"Run workflow"**

## 📊 Pipeline Metrics

You can monitor:
- **Success Rate**: Percentage of successful runs
- **Runtime**: How long each job takes
- **Data Collection**: Amount of data collected per run
- **Model Performance**: R² scores and MAE for each model

## 🔒 Security Notes

- API keys are stored as encrypted secrets
- Never commit sensitive data to repository
- Use GitHub's built-in security features

## 💡 Best Practices

1. **Monitor Regularly**: Check pipeline status weekly
2. **Review Logs**: Examine failed runs for issues
3. **Update Dependencies**: Keep requirements.txt current
4. **Test Locally**: Verify changes before pushing
5. **Backup Models**: Download important model artifacts

## 📞 Support

If you encounter issues:
1. Check GitHub Actions logs for detailed error messages
2. Verify all secrets are configured correctly
3. Ensure repository permissions allow Actions
4. Test individual components locally first

---

**🎯 Your Karachi AQI prediction system is now fully automated with GitHub Actions!**
