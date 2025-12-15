# Elliott Wave Predictor - Easy Launcher

This repository includes a simple launcher to easily start the Elliott Wave Predictor system.

## Quick Start

### Method 1: Using the Python Launcher (Recommended)
```bash
# Make sure you're in the project directory
cd /home/pc/ElliotWavePredictor

# Activate virtual environment
source venv/bin/activate

# Run the launcher
python3 start_system.py
```

### Method 2: Using the Shell Script
```bash
# Make sure you're in the project directory
cd /home/pc/ElliotWavePredictor

# Make the script executable (if not already)
chmod +x start.sh

# Run the script
./start.sh
```

## Launcher Options

The launcher accepts several command-line options:

```bash
# Specify a different port (default: 8000)
python3 start_system.py --port 8005

# Specify a different host (default: 127.0.0.1)
python3 start_system.py --host 0.0.0.0

# Skip pre-cleaning operations
python3 start_system.py --no-clean

# Skip dependency checking
python3 start_system.py --skip-deps
```

## Accessing the System

Once the backend is running:

1. Open Firefox (or any web browser)
2. Navigate to `http://127.0.0.1:8000` (or the port you specified)
3. The frontend will automatically connect to the backend
4. You can interact with the Elliott Wave analysis tools

## Stopping the System

Press `Ctrl+C` in the terminal where the launcher is running to stop the backend server.

## What the Launcher Does

1. **Pre-Cleaning**: Removes temporary files and databases that might cause conflicts
2. **Dependency Check**: Verifies all required Python packages are installed
3. **Backend Launch**: Starts the FastAPI backend server
4. **Post-Cleaning**: Cleans up temporary files when shutting down

## Troubleshooting

### Port Already in Use
If you see an error about a port being in use, try:
```bash
python3 start_system.py --port 8005
```

### Missing Dependencies
If dependencies are missing, install them with:
```bash
source venv/bin/activate
pip install pandas numpy fastapi uvicorn[standard] scikit-learn joblib
```

### Backend Won't Start
Check that:
1. The virtual environment is activated
2. All dependencies are installed
3. No other process is using the specified port