"""
MongoDB Feature Store for Karachi AQI Data
==========================================

A MongoDB-based feature store implementation for storing and retrieving
Karachi AQI data. Supports both local MongoDB and MongoDB Atlas.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import motor.motor_asyncio
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBFeatureStore:
    """
    MongoDB-based feature store for Karachi AQI data
    """

    def __init__(self, connection_string: str = "mongodb://localhost:27017/",
                 database_name: str = "karachi_aqi_db",
                 feature_group_name: str = "karachi_aqi_features"):
        """
        Initialize MongoDB feature store

        Args:
            connection_string: MongoDB connection string
            database_name: Database name
            feature_group_name: Collection name for features
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.feature_group_name = feature_group_name

        # Initialize clients
        self.sync_client = None
        self.async_client = None
        self.database = None
        self.collection = None

        # Connect to MongoDB
        self._connect()

        # Create indexes for better performance
        self._create_indexes()

    def _connect(self):
        """Connect to MongoDB"""
        try:
            # Synchronous client
            self.sync_client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.sync_client.admin.command('ping')
            logger.info("✅ Connected to MongoDB (synchronous)")

            # Asynchronous client
            self.async_client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_string)
            logger.info("✅ Connected to MongoDB (asynchronous)")

            # Get database and collection
            self.database = self.sync_client[self.database_name]
            self.collection = self.database[self.feature_group_name]

            logger.info(f"📊 Using database: {self.database_name}")
            logger.info(f"📋 Using collection: {self.feature_group_name}")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            logger.info("💡 For local MongoDB, install MongoDB Community Server")
            logger.info("💡 For cloud MongoDB, use MongoDB Atlas connection string")
            logger.info("📖 Falling back to CSV-based storage for now")

            # Fallback to CSV-based storage
            self._fallback_to_csv()
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to MongoDB: {e}")
            self._fallback_to_csv()

    def _create_indexes(self):
        """Create indexes for better query performance"""
        try:
            if self.collection is not None:
                # Index on timestamp for time-based queries
                self.collection.create_index([("timestamp", ASCENDING)], name="timestamp_index")

                # Index on city for city-based filtering
                self.collection.create_index([("city", ASCENDING)], name="city_index")

                # Compound index for city and timestamp
                self.collection.create_index([
                    ("city", ASCENDING),
                    ("timestamp", ASCENDING)
                ], name="city_timestamp_index")

                logger.info("✅ Created MongoDB indexes for optimal performance")
        except Exception as e:
            logger.warning(f"⚠️ Could not create indexes: {e}")

    def _fallback_to_csv(self):
        """Fallback to CSV-based storage when MongoDB is not available"""
        logger.info("📁 Using CSV-based feature store as fallback")

        self.data_dir = Path("../data")
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_file = self.data_dir / "karachi_raw_data.csv"
        self.metadata_file = self.data_dir / "karachi_metadata.json"

        # Initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Load CSV metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'city': 'Karachi',
                'country': 'Pakistan',
                'total_records': 0,
                'last_updated': None,
                'data_quality': {},
                'features': []
            }

    def _save_metadata(self):
        """Save CSV metadata"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def insert_data(self, data_df: pd.DataFrame) -> bool:
        """
        Insert data into the feature store

        Args:
            data_df: DataFrame with Karachi AQI data

        Returns:
            bool: Success status
        """
        try:
            # Validate data
            required_columns = ['timestamp', 'temperature', 'humidity', 'aqi']
            missing_columns = [col for col in required_columns if col not in data_df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Ensure city column exists
            if 'city' not in data_df.columns:
                data_df = data_df.copy()
                data_df['city'] = 'Karachi'

            # Convert DataFrame to dictionary format for MongoDB
            records = data_df.to_dict('records')

            # Convert timestamps to datetime objects
            for record in records:
                if isinstance(record['timestamp'], str):
                    record['timestamp'] = pd.to_datetime(record['timestamp'])
                elif isinstance(record['timestamp'], (int, float)):
                    record['timestamp'] = datetime.fromtimestamp(record['timestamp'])

            # Insert into MongoDB or CSV
            if self.collection is not None:
                # MongoDB insertion
                result = self.collection.insert_many(records)
                inserted_count = len(result.inserted_ids)

                logger.info(f"✅ Inserted {inserted_count} records into MongoDB")

                # Update metadata
                total_records = self.collection.count_documents({})
                self._update_mongodb_metadata(total_records)

            else:
                # CSV fallback
                self._insert_csv_data(data_df)

            return True

        except Exception as e:
            logger.error(f"❌ Error inserting data: {e}")
            return False

    def _insert_csv_data(self, data_df: pd.DataFrame):
        """Insert data into CSV file (fallback)"""
        try:
            # Load existing data if available
            existing_data = pd.DataFrame()
            if self.raw_data_file.exists():
                existing_data = pd.read_csv(self.raw_data_file)
                if 'timestamp' in existing_data.columns:
                    existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])

            # Combine with new data
            combined_data = pd.concat([existing_data, data_df], ignore_index=True)

            # Remove duplicates based on timestamp
            combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='last')

            # Sort by timestamp
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)

            # Save to CSV
            combined_data.to_csv(self.raw_data_file, index=False)

            # Update metadata
            self.metadata['total_records'] = len(combined_data)
            self.metadata['features'] = list(combined_data.columns)
            self.metadata['data_quality'] = {
                'missing_values': combined_data.isnull().sum().to_dict(),
                'date_range': {
                    'start': combined_data['timestamp'].min().isoformat() if not combined_data.empty else None,
                    'end': combined_data['timestamp'].max().isoformat() if not combined_data.empty else None
                }
            }
            self._save_metadata()

            logger.info(f"💾 Inserted {len(data_df)} records into CSV feature store")

        except Exception as e:
            logger.error(f"❌ Error inserting CSV data: {e}")

    def _update_mongodb_metadata(self, total_records: int):
        """Update MongoDB metadata"""
        try:
            # Get data quality info
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "min_timestamp": {"$min": "$timestamp"},
                        "max_timestamp": {"$max": "$timestamp"},
                        "total_records": {"$sum": 1}
                    }
                }
            ]

            result = list(self.collection.aggregate(pipeline))
            if result:
                metadata = result[0]
                # Store metadata in a separate collection
                metadata_collection = self.database['metadata']
                metadata_collection.update_one(
                    {"feature_group": self.feature_group_name},
                    {
                        "$set": {
                            "feature_group": self.feature_group_name,
                            "city": "Karachi",
                            "country": "Pakistan",
                            "total_records": total_records,
                            "last_updated": datetime.now(),
                            "date_range": {
                                "start": metadata.get("min_timestamp"),
                                "end": metadata.get("max_timestamp")
                            }
                        }
                    },
                    upsert=True
                )

        except Exception as e:
            logger.warning(f"⚠️ Could not update MongoDB metadata: {e}")

    def get_recent_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Get recent data from feature store

        Args:
            hours: Number of hours of data to retrieve

        Returns:
            DataFrame with recent data
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            if self.collection is not None:
                # MongoDB query
                query = {
                    "timestamp": {"$gte": cutoff_time},
                    "city": "Karachi"
                }

                cursor = self.collection.find(query).sort("timestamp", DESCENDING)
                records = list(cursor)

                if records:
                    df = pd.DataFrame(records)
                    # Remove MongoDB _id column
                    if '_id' in df.columns:
                        df = df.drop('_id', axis=1)
                    logger.info(f"✅ Retrieved {len(df)} recent records from MongoDB")
                    return df
                else:
                    logger.warning("⚠️ No recent data found in MongoDB")
                    return pd.DataFrame()

            else:
                # CSV fallback
                return self._get_csv_recent_data(hours)

        except Exception as e:
            logger.error(f"❌ Error retrieving recent data: {e}")
            return pd.DataFrame()

    def _get_csv_recent_data(self, hours: int) -> pd.DataFrame:
        """Get recent data from CSV file"""
        try:
            if not self.raw_data_file.exists():
                logger.warning("⚠️ No CSV data file found")
                return pd.DataFrame()

            data = pd.read_csv(self.raw_data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Filter by time range
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = data[data['timestamp'] >= cutoff_time]

            logger.info(f"✅ Retrieved {len(recent_data)} recent records from CSV")
            return recent_data

        except Exception as e:
            logger.error(f"❌ Error retrieving CSV recent data: {e}")
            return pd.DataFrame()

    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get historical data for model training

        Args:
            days: Number of days of historical data

        Returns:
            DataFrame with historical data
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            if self.collection is not None:
                # MongoDB query
                query = {
                    "timestamp": {"$gte": cutoff_date},
                    "city": "Karachi"
                }

                cursor = self.collection.find(query).sort("timestamp", ASCENDING)
                records = list(cursor)

                if records:
                    df = pd.DataFrame(records)
                    if '_id' in df.columns:
                        df = df.drop('_id', axis=1)
                    logger.info(f"✅ Retrieved {len(df)} historical records from MongoDB")
                    return df
                else:
                    logger.warning("⚠️ No historical data found in MongoDB")
                    return pd.DataFrame()

            else:
                # CSV fallback
                return self._get_csv_historical_data(days)

        except Exception as e:
            logger.error(f"❌ Error retrieving historical data: {e}")
            return pd.DataFrame()

    def _get_csv_historical_data(self, days: int) -> pd.DataFrame:
        """Get historical data from CSV file"""
        try:
            if not self.raw_data_file.exists():
                logger.warning("⚠️ No CSV data file found")
                return pd.DataFrame()

            data = pd.read_csv(self.raw_data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            historical_data = data[data['timestamp'] >= cutoff_date]

            logger.info(f"✅ Retrieved {len(historical_data)} historical records from CSV")
            return historical_data

        except Exception as e:
            logger.error(f"❌ Error retrieving CSV historical data: {e}")
            return pd.DataFrame()

    def get_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        try:
            if self.collection is not None:
                # MongoDB stats
                total_records = self.collection.count_documents({"city": "Karachi"})

                # Get date range
                pipeline = [
                    {"$match": {"city": "Karachi"}},
                    {
                        "$group": {
                            "_id": None,
                            "min_timestamp": {"$min": "$timestamp"},
                            "max_timestamp": {"$max": "$timestamp"}
                        }
                    }
                ]

                result = list(self.collection.aggregate(pipeline))
                date_range = {}
                if result:
                    date_range = {
                        "start": result[0].get("min_timestamp"),
                        "end": result[0].get("max_timestamp")
                    }

                # Get AQI statistics
                aqi_pipeline = [
                    {"$match": {"city": "Karachi"}},
                    {
                        "$group": {
                            "_id": None,
                            "avg_aqi": {"$avg": "$aqi"},
                            "min_aqi": {"$min": "$aqi"},
                            "max_aqi": {"$max": "$aqi"},
                            "count": {"$sum": 1}
                        }
                    }
                ]

                aqi_result = list(self.collection.aggregate(aqi_pipeline))
                aqi_stats = {}
                if aqi_result:
                    aqi_stats = {
                        "mean": aqi_result[0].get("avg_aqi"),
                        "min": aqi_result[0].get("min_aqi"),
                        "max": aqi_result[0].get("max_aqi")
                    }

                return {
                    'total_records': total_records,
                    'date_range': date_range,
                    'aqi_stats': aqi_stats,
                    'storage_type': 'MongoDB'
                }

            else:
                # CSV stats
                if self.raw_data_file.exists():
                    data = pd.read_csv(self.raw_data_file)
                    return {
                        'total_records': len(data),
                        'features_count': len(data.columns),
                        'storage_type': 'CSV',
                        'aqi_stats': {
                            'mean': float(data['aqi'].mean()) if 'aqi' in data.columns else None,
                            'min': float(data['aqi'].min()) if 'aqi' in data.columns else None,
                            'max': float(data['aqi'].max()) if 'aqi' in data.columns else None
                        }
                    }
                else:
                    return {'total_records': 0, 'storage_type': 'CSV'}

        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {'error': str(e)}

    def clear_data(self) -> bool:
        """Clear all data from feature store"""
        try:
            if self.collection is not None:
                # Clear MongoDB collection
                result = self.collection.delete_many({"city": "Karachi"})
                logger.info(f"🗑️ Cleared {result.deleted_count} records from MongoDB")

                # Clear metadata
                metadata_collection = self.database['metadata']
                metadata_collection.delete_many({"feature_group": self.feature_group_name})

            else:
                # Clear CSV files
                if self.raw_data_file.exists():
                    self.raw_data_file.unlink()
                if self.metadata_file.exists():
                    self.metadata_file.unlink()

                self.metadata = {
                    'city': 'Karachi',
                    'country': 'Pakistan',
                    'total_records': 0,
                    'last_updated': None,
                    'data_quality': {},
                    'features': []
                }

            logger.info("✅ Feature store cleared")
            return True

        except Exception as e:
            logger.error(f"❌ Error clearing data: {e}")
            return False

    def close_connection(self):
        """Close MongoDB connection"""
        if self.sync_client:
            self.sync_client.close()
            logger.info("🔌 MongoDB connection closed")

if __name__ == "__main__":
    # Test the MongoDB feature store
    print("🧪 Testing MongoDB Feature Store")
    print("=" * 50)

    # Try to connect to MongoDB
    store = MongoDBFeatureStore()

    # Generate sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=10), periods=10, freq='H'),
        'city': ['Karachi'] * 10,
        'temperature': np.random.normal(30, 3, 10),
        'humidity': np.random.normal(65, 8, 10),
        'aqi': np.random.normal(100, 20, 10),
        'pm2_5': np.random.normal(45, 15, 10)
    })

    # Test insertion
    print("\n📥 Testing data insertion...")
    success = store.insert_data(sample_data)
    if success:
        print("✅ Data insertion successful")
    else:
        print("❌ Data insertion failed")

    # Test retrieval
    print("\n📤 Testing data retrieval...")
    recent_data = store.get_recent_data(hours=12)
    print(f"✅ Retrieved {len(recent_data)} recent records")

    historical_data = store.get_historical_data(days=1)
    print(f"✅ Retrieved {len(historical_data)} historical records")

    # Test statistics
    print("\n📊 Testing statistics...")
    stats = store.get_statistics()
    print(f"✅ Statistics: {stats}")

    # Close connection
    store.close_connection()
    print("\n🎉 MongoDB Feature Store test completed!")


