# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

This is an Elliott Wave Predictor - a local, web-based tool for analyzing financial data using Elliott Wave theory. The backend is powered by Python, and the frontend is built with vanilla JavaScript, HTML, and CSS.

## Repository Structure

- `analysis/`: Core Elliott Wave analysis logic including wave detection, confidence scoring, and non-maximum suppression
- `backend/`: Main Python backend server (`main.py`) that serves the API and WebSocket endpoints
- `data/`: Sample financial data in CSV format
- `docs/`: Documentation about academic sources for Elliott Wave theory
- `frontend/`: Client-side JavaScript, HTML, and CSS files
- `ingest/`: Data ingestion adapters for various financial data sources
- `tests/`: Python test files for various components

## Dependencies

Install Python dependencies:
```bash
pip install pytest pandas numpy fastapi uvicorn[standard] python-multipart
```

Note: Some components may require additional dependencies depending on the data sources used.

## Development Commands

### Running the Application

1. Start the backend server:
   ```bash
   python backend/main.py
   ```

2. Open the frontend:
   Open `frontend/index.html` in your web browser.

### Testing

Tests are written using pytest. Run all tests with:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_wave_rules.py
```

Individual test dependencies:
- Basic tests: `pip install pytest`
- Analysis tests: `pip install pytest pandas numpy`
- Rate limiting tests: `pip install pytest fastapi uvicorn[standard] python-multipart`

## Code Architecture

### Backend Architecture

The backend (`backend/main.py`) serves as the main server, providing:
- REST API endpoints for data ingestion and retrieval
- WebSocket connections for real-time updates to the frontend
- Integration with data adapters in `ingest/adapters.py`
- Database management with automatic TTL cleanup

Key components:
- Data ingestion from various sources (Binance, Yahoo Finance, etc.)
- SQLite database storage with TTL-based cleanup
- Rate limiting for API endpoints
- Real-time WebSocket communication with frontend

### Analysis Engine

Core analysis logic is in the `analysis/` directory:
- `wave_detector.py`: Implements Elliott Wave pattern detection algorithms
- `confidence.py`: Calculates confidence scores for detected wave patterns
- `nms.py`: Applies non-maximum suppression to filter overlapping wave candidates
- `elliott_rules.py`: Implements Elliott Wave rule validation

### Frontend Architecture

The frontend (`frontend/`) consists of:
- `index.html`: Main HTML structure
- `app.js`: Client-side JavaScript handling UI interactions, WebSocket communication, and chart rendering
- `styles.css`: Styling for the application

Key features:
- Real-time price chart visualization using Canvas
- WebSocket connection to backend for live updates
- Interactive controls for symbol selection and interval setting
- Display of wave pattern candidates and confidence scores

### Data Flow

1. User selects a financial symbol and interval in the frontend
2. Frontend sends request to backend API
3. Backend retrieves data from configured data source via adapters
4. Data is stored in SQLite database with TTL
5. Analysis engine processes data to detect Elliott Wave patterns
6. Results are sent to frontend via WebSocket
7. Frontend renders charts and displays confidence scores