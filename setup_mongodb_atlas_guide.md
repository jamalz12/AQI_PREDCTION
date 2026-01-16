# 🗄️ MongoDB Atlas Setup Guide for Karachi AQI Feature Store

This guide shows you how to set up MongoDB Atlas (cloud MongoDB) to replace the CSV fallback with a real database feature store, similar to Hopsworks.

## 📋 Prerequisites

- MongoDB Atlas account (free tier available)
- Internet connection for cloud database access

## 🚀 Step-by-Step Setup

### Step 1: Create MongoDB Atlas Account

1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Click "Try Free" or "Sign Up"
3. Create your account and verify your email

### Step 2: Create a Free Cluster

1. After login, click "Build a Database"
2. Choose "M0 Cluster" (Free tier)
3. Select your preferred cloud provider and region
4. Choose cluster name (e.g., "KarachiAQI")
5. Click "Create Cluster"

### Step 3: Set Up Database Access

1. Go to "Database Access" in the left sidebar
2. Click "Add New Database User"
3. Choose "Password" authentication
4. Enter username: `karachi_aqi_user`
5. Enter a strong password (save this!)
6. Set user privileges to "Read and write to any database"
7. Click "Add User"

### Step 4: Configure Network Access

1. Go to "Network Access" in the left sidebar
2. Click "Add IP Address"
3. Choose "Allow Access from Anywhere" (0.0.0.0/0)
4. Click "Confirm"

### Step 5: Get Connection String

1. Go to "Clusters" in the left sidebar
2. Click "Connect" on your cluster
3. Choose "Connect your application"
4. Copy the connection string, it will look like:
   ```
   mongodb+srv://karachi_aqi_user:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

5. Replace `<password>` with your actual password

### Step 6: Update Your Code

Update the MongoDB connection string in your `karachi_aqi_app.py`:

```python
# Replace this line:
mongo_connection = "mongodb://localhost:27017/"  # Default local MongoDB

# With your Atlas connection string:
mongo_connection = "mongodb+srv://karachi_aqi_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
```

## 🧪 Testing Your MongoDB Atlas Connection

### Test 1: Connection Test

Create a test file to verify your connection:

```python
from mongodb_feature_store import MongoDBFeatureStore

# Test with your Atlas connection string
atlas_connection = "mongodb+srv://karachi_aqi_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"

store = MongoDBFeatureStore(connection_string=atlas_connection)
stats = store.get_statistics()
print("✅ MongoDB Atlas connection successful!" if 'total_records' in stats else "❌ Connection failed")
```

### Test 2: Data Operations

```python
import pandas as pd
from datetime import datetime

# Test data insertion
test_data = pd.DataFrame({
    'timestamp': [datetime.now()],
    'city': ['Karachi'],
    'temperature': [32.5],
    'humidity': [65.0],
    'aqi': [120.0],
    'pm2_5': [55.0]
})

success = store.insert_data(test_data)
print("✅ Data insertion successful!" if success else "❌ Data insertion failed")

# Test data retrieval
recent_data = store.get_recent_data(hours=1)
print(f"✅ Retrieved {len(recent_data)} records")
```

## 🔧 Configuration Options

### Database and Collection Names

You can customize the database and collection names:

```python
store = MongoDBFeatureStore(
    connection_string="your_atlas_connection_string",
    database_name="karachi_aqi_db",        # Your database name
    feature_group_name="karachi_features"   # Your collection name
)
```

### Environment Variables (Recommended for Production)

Create a `.env` file:

```env
MONGODB_ATLAS_CONNECTION=mongodb+srv://karachi_aqi_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=karachi_aqi_db
COLLECTION_NAME=karachi_features
```

Then use in code:

```python
import os
from dotenv import load_dotenv

load_dotenv()

store = MongoDBFeatureStore(
    connection_string=os.getenv('MONGODB_ATLAS_CONNECTION'),
    database_name=os.getenv('DATABASE_NAME', 'karachi_aqi_db'),
    feature_group_name=os.getenv('COLLECTION_NAME', 'karachi_features')
)
```

## 📊 MongoDB Atlas Features for AQI Data

### Advantages over CSV Storage:

1. **Scalability**: Handle millions of AQI records
2. **Performance**: Fast queries with indexing
3. **Reliability**: Cloud-based with automatic backups
4. **Real-time**: Support for real-time data streaming
5. **Analytics**: Built-in aggregation and analytics
6. **Security**: Enterprise-grade security features

### Automatic Features:

- **Indexing**: Automatic indexing on timestamp and city fields
- **Metadata**: Automatic metadata collection and updates
- **Aggregation**: Built-in statistical aggregations
- **Backup**: Automatic cloud backups

## 🔍 Monitoring Your Database

### Atlas Dashboard

1. Go to your Atlas dashboard
2. View real-time metrics:
   - Connection count
   - Operation count
   - Data size
   - Index performance

### Query Analytics

1. Go to "Performance" tab
2. View slow queries
3. Optimize indexes based on query patterns

## 🛠️ Troubleshooting

### Common Issues:

#### Connection Timeout
- Check your IP whitelist (Network Access)
- Verify username/password
- Ensure cluster is in "running" state

#### Authentication Failed
- Double-check username and password
- Ensure user has "Read and write" permissions

#### Database Not Found
- The database is created automatically on first write
- No need to manually create databases in MongoDB

## 💰 Cost Information

### Free Tier (M0)
- **Storage**: 512 MB
- **Connections**: 500 max
- **Data Transfer**: 512 MB/month

### Upgrade Options
- **M2**: $0.08/hour (~$60/month)
- **M5**: $0.62/hour (~$450/month)

For AQI data, the free tier should be sufficient for several years of data.

## 🔄 Switching from CSV to MongoDB

Your app automatically detects and uses MongoDB when available, falling back to CSV when MongoDB is not accessible. No code changes needed!

## 🎯 Next Steps

1. **Set up MongoDB Atlas** (follow steps above)
2. **Update connection string** in `karachi_aqi_app.py`
3. **Test the connection** using the test scripts
4. **Enjoy cloud-scale feature storage!**

## 📞 Support

- [MongoDB Atlas Documentation](https://docs.mongodb.com/atlas/)
- [MongoDB University](https://university.mongodb.com/) (free courses)
- [MongoDB Community Forums](https://community.mongodb.com/)

---

**🎉 Your Karachi AQI system now has enterprise-grade feature storage with MongoDB Atlas!**


