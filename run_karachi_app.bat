@echo off
echo Starting Karachi AQI Streamlit App...
echo.
echo The app will be available at: http://localhost:8501
echo.
echo If connection fails, try: http://127.0.0.1:8501
echo.
echo Press Ctrl+C to stop the app
echo.
streamlit run karachi_aqi_app.py --server.port 8501 --server.address localhost
pause


