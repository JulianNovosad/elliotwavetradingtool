# Elliott Wave Predictor

A local, web-based tool for analyzing financial data using Elliott Wave theory.

## Description

This tool provides a web interface to visualize and analyze financial time-series data, with a focus on identifying Elliott Wave patterns. The backend is powered by Python, and the frontend is built with vanilla JavaScript, HTML, and CSS.

## Getting Started

### Prerequisites

- Python 3.x
- Node.js and npm

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ElliotWavePredictor
    ```

2.  **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install frontend dependencies:**
    ```bash
    npm install
    ```

### Running the Application

1.  **Start the backend server:**
    ```bash
    python backend/main.py
    ```

2.  **Open the frontend:**
    Open the `frontend/index.html` file in your web browser.

## Usage

-   Load financial data in CSV format.
-   The tool will automatically analyze the data and display potential Elliott Wave patterns.
-   The confidence of the detected waves will be displayed.
