# Elliott Wave Predictor

This is a local, web-based tool for analyzing financial data using Elliott Wave theory.

## Project Overview

The project is a web application with a Python backend and a vanilla JavaScript frontend. The backend is built with FastAPI and serves the frontend, which is composed of HTML, CSS, and JavaScript. The backend also provides a WebSocket for real-time analysis updates. The core of the project is the Elliott Wave analysis pipeline, which is responsible for detecting and analyzing Elliott Wave patterns in financial time-series data.

**Key Technologies:**

*   **Backend:** Python, FastAPI, SQLite, pandas, numpy
*   **Frontend:** JavaScript, HTML, CSS
*   **Data Analysis:** The analysis pipeline uses a combination of rule-based systems and machine learning to detect and score Elliott Wave patterns. The pipeline is composed of the following modules:
    *   `WaveDetector`: Orchestrates the analysis pipeline.
    *   `ElliottRuleEngine`: Applies Elliott Wave rules to validate wave patterns.
    *   `NMS` (Non-Maximum Suppression): Removes overlapping wave segments.
    *   `ConfidenceScorer`: Scores the confidence of detected waves.
    *   `TechnicalIndicators`: Calculates various technical indicators.
    *   `MLPredictor`: A machine learning model for making predictions (currently a placeholder).

## Building and Running

### Prerequisites

*   Python 3.x
*   uvicorn
*   fastapi
*   pandas
*   pyyaml
*   websockets
*   aiohttp
*   python-binance
*   yfinance
*   scipy
*   numpy


### Running the Application

1.  **Start the backend server:**

    ```bash
    python backend/main.py
    ```

2.  **Open the frontend:**

    Open `http://localhost:8003` in your web browser.

## Development Conventions

*   The backend code is located in the `backend` directory.
*   The frontend code is located in the `frontend` directory.
*   The data analysis code is located in the `analysis` directory.
*   The project uses a `config.yaml` file for configuration.
*   The project uses a SQLite database to store price data.
*   The wave detection and candidate generation logic in `analysis/wave_detector.py` is currently using mock data and is under development.
