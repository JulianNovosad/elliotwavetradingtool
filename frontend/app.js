// frontend/app.js

// --- DOM Elements ---
const symbolInput = document.getElementById('symbol-input');
const intervalSelect = document.getElementById('interval-select');
const sourceSelect = document.getElementById('source-select');
const setSymbolBtn = document.getElementById('set-symbol-btn');
const priceChartCanvas = document.getElementById('price-chart');
const currentTimeDisplay = document.getElementById('current-time');
const currentPriceDisplay = document.getElementById('current-price');
const confidenceScoresDiv = document.getElementById('confidence-scores');
const waveCandidatesDiv = document.getElementById('wave-candidates');
const technicalIndicatorsDiv = document.getElementById('technical-indicators');
const mlPredictionsDiv = document.getElementById('ml-predictions');

// --- Constants and Configuration ---
const WS_URL = `ws://${window.location.host}/ws`; // Backend WebSocket URL
const API_BASE_URL = `http://${window.location.host}`; // Backend API base URL

const CANVAS_HEIGHT = 400; // Matches CSS height for chart-panel canvas
const CHART_PADDING = 30; // Padding around the chart drawing area

// Colors as specified:
// Background: white (handled by CSS)
// Candles/Price/Lines: black
const COLOR_BLACK = '#000000';
const COLOR_WHITE = '#FFFFFF';
// Elliott waves: red when price goes down, green when price goes up
const COLOR_WAVE_UP = '#00FF00'; // Green
const COLOR_WAVE_DOWN = '#FF0000'; // Red

const MAX_PRICES_TO_DISPLAY = 500; // Limit the number of price points shown for performance

// Chart scaling and data storage
let chartData = {
    prices: [], // Array of {t: timestamp, p: price}
    wave_levels: [], // Array of wave levels, each with segments
    candidates: [], // Array of top wave count candidates
    symbol: '',
    timestamp: '', // Latest frame timestamp
    current_price: null,
};

// Global chart context and rendering state
let ctx;
let chartWidth, chartHeight;
let priceScaleY, timeScaleX;

// --- WebSocket Connection ---
let websocket;

function connectWebSocket() {
    websocket = new WebSocket(WS_URL);

    websocket.onopen = () => {
        console.log('WebSocket connected');
        // Optionally send initial symbol if already set or load default
    };

    websocket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            console.log('Received WebSocket message:', message);
            updateChartData(message);
            renderChart();
            updateUI(message);
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
        }
    };

    websocket.onclose = () => {
        console.log('WebSocket disconnected');
        // Attempt to reconnect after a delay
        setTimeout(connectWebSocket, 5000); // Reconnect every 5 seconds
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// --- Data Handling ---
function updateChartData(message) {
    chartData.symbol = message.symbol || chartData.symbol;
    chartData.timestamp = message.timestamp || chartData.timestamp;
    chartData.current_price = message.prices && message.prices.length > 0 ? message.prices[message.prices.length - 1].p : chartData.current_price;
    
    // Update prices, keeping only the last MAX_PRICES_TO_DISPLAY
    if (message.prices) {
        chartData.prices = message.prices;
        if (chartData.prices.length > MAX_PRICES_TO_DISPLAY) {
            chartData.prices = chartData.prices.slice(-MAX_PRICES_TO_DISPLAY);
        }
    }
    
    chartData.wave_levels = message.wave_levels || [];
    chartData.candidates = message.candidates || [];
    chartData.technical_indicators = message.technical_indicators || {};
    chartData.ml_predictions = message.ml_predictions || [];
    
    // Ensure price data is sorted by timestamp for correct rendering
    chartData.prices.sort((a, b) => new Date(a.t) - new Date(b.t));
}

// --- UI Updates ---
function updateUI(data) {
    // Update time and price display
    if (data.timestamp) {
        const date = new Date(data.timestamp);
        currentTimeDisplay.textContent = date.toLocaleString(); // User's locale format
    }
    if (data.current_price !== null) {
        currentPriceDisplay.textContent = data.current_price.toFixed(2); // Display with 2 decimal places
    }

    // Update confidence breakdown panel
    confidenceScoresDiv.innerHTML = ''; // Clear previous content
    if (data.candidates && data.candidates.length > 0) {
        data.candidates.forEach(candidate => {
            const p = document.createElement('p');
            p.innerHTML = `<strong>${candidate.rank}.</strong> Confidence: ${candidate.confidence.toFixed(3)} | ${candidate.description}`;
            confidenceScoresDiv.appendChild(p);
        });
    } else {
        confidenceScoresDiv.innerHTML = '<p>No wave count candidates available.</p>';
    }

    // Update wave candidates panel (can be same as confidence breakdown or more detailed)
    waveCandidatesDiv.innerHTML = ''; // Clear previous content
    if (data.wave_levels && data.wave_levels.length > 0) {
        data.wave_levels.forEach(level => {
            const levelDiv = document.createElement('div');
            levelDiv.innerHTML = `<h4>Level ${level.level} (${level.segments.length} segments)</h4>`;
            
            const segmentsList = document.createElement('ul');
            segmentsList.style.listStyle = 'disc';
            segmentsList.style.paddingLeft = '20px';
            
            level.segments.forEach(segment => {
                const li = document.createElement('li');
                const direction = segment.direction === 'up' ? '<span style="color:green;">↑</span>' : '<span style="color:red;">↓</span>';
                li.innerHTML = `[${new Date(segment.start).toLocaleTimeString()} - ${new Date(segment.end).toLocaleTimeString()}] ${segment.label || ''} (${direction}) Conf: ${segment.confidence.toFixed(3)}`;
                segmentsList.appendChild(li);
            });
            levelDiv.appendChild(segmentsList);
            waveCandidatesDiv.appendChild(levelDiv);
        });
    } else {
        waveCandidatesDiv.innerHTML = '<p>No wave levels detected.</p>';
    }
    
    // Update technical indicators panel
    technicalIndicatorsDiv.innerHTML = ''; // Clear previous content
    if (data.technical_indicators && Object.keys(data.technical_indicators).length > 0) {
        // Display some key indicators
        const indicatorsToShow = ['rsi', 'macd_line', 'macd_signal', 'sma_20', 'sma_50'];
        
        indicatorsToShow.forEach(indicatorName => {
            if (data.technical_indicators[indicatorName]) {
                // Get the latest value (last element)
                const indicatorValues = data.technical_indicators[indicatorName];
                if (Array.isArray(indicatorValues) && indicatorValues.length > 0) {
                    const latestValue = indicatorValues[indicatorValues.length - 1];
                    const p = document.createElement('p');
                    p.innerHTML = `<strong>${indicatorName.toUpperCase()}:</strong> ${typeof latestValue === 'number' ? latestValue.toFixed(2) : latestValue}`;
                    technicalIndicatorsDiv.appendChild(p);
                }
            }
        });
        
        // If no indicators were shown, display a message
        if (technicalIndicatorsDiv.innerHTML === '') {
            technicalIndicatorsDiv.innerHTML = '<p>Technical indicators calculated but none to display.</p>';
        }
    } else {
        technicalIndicatorsDiv.innerHTML = '<p>No technical indicators available.</p>';
    }
    
    // Update ML predictions panel
    mlPredictionsDiv.innerHTML = ''; // Clear previous content
    if (data.ml_predictions && data.ml_predictions.length > 0) {
        data.ml_predictions.forEach(prediction => {
            const p = document.createElement('p');
            const predictionText = prediction.prediction === 1 ? 'UP' : 'DOWN';
            const predictionColor = prediction.prediction === 1 ? 'green' : 'red';
            p.innerHTML = `
                <strong>Prediction:</strong> 
                <span style="color: ${predictionColor}; font-weight: bold;">${predictionText}</span> | 
                <strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(1)}% | 
                <strong>Probability Up:</strong> ${(prediction.probability_up * 100).toFixed(1)}%
            `;
            mlPredictionsDiv.appendChild(p);
        });
    } else {
        mlPredictionsDiv.innerHTML = '<p>No ML predictions available.</p>';
    }
}

// --- Chart Rendering ---
function renderChart() {
    if (!ctx || !chartData.prices || chartData.prices.length === 0) {
        // console.log("Not enough data or canvas not ready to render.");
        return;
    }

    // Clear canvas
    ctx.clearRect(0, 0, chartWidth, chartHeight);

    // Set drawing styles
    ctx.lineWidth = 1;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Determine scales
    const timestamps = chartData.prices.map(p => new Date(p.t));
    const prices = chartData.prices.map(p => p.p);

    const minTime = timestamps[0].getTime();
    const maxTime = timestamps[timestamps.length - 1].getTime();
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    // Calculate drawing area dimensions
    const drawingWidth = chartWidth - 2 * CHART_PADDING;
    const drawingHeight = chartHeight - 2 * CHART_PADDING;

    // Price scale Y: Y = drawingHeight - (price - minPrice) / (maxPrice - minPrice) * drawingHeight
    // Time scale X: X = CHART_PADDING + (timestamp - minTime) / (maxTime - minTime) * drawingWidth

    const priceRange = maxPrice - minPrice;
    const timeRange = maxTime - minTime;

    if (priceRange === 0 || timeRange === 0) {
        console.warn("Price or time range is zero, cannot scale chart.");
        return;
    }

    // Draw X and Y axes (black lines)
    ctx.strokeStyle = COLOR_BLACK;
    // Y-axis (price scale)
    ctx.beginPath();
    ctx.moveTo(CHART_PADDING, CHART_PADDING);
    ctx.lineTo(CHART_PADDING, chartHeight - CHART_PADDING);
    ctx.stroke();
    // X-axis (time scale)
    ctx.beginPath();
    ctx.moveTo(CHART_PADDING, chartHeight - CHART_PADDING);
    ctx.lineTo(chartWidth - CHART_PADDING, chartHeight - CHART_PADDING);
    ctx.stroke();

    // Draw price scale labels (e.g., every 10% of range)
    ctx.fillStyle = COLOR_BLACK;
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i++) {
        const price = minPrice + (priceRange * (i / 10));
        const y = chartHeight - CHART_PADDING - (i / 10) * drawingHeight;
        ctx.fillText(price.toFixed(2), CHART_PADDING - 5, y);
    }

    // Draw time scale labels (e.g., every few points)
    ctx.textAlign = 'center';
    const timeLabelInterval = Math.max(1, Math.floor(chartData.prices.length / 10)); // Show about 10 labels
    for (let i = 0; i < chartData.prices.length; i += timeLabelInterval) {
        const timestamp = new Date(chartData.prices[i].t);
        const x = CHART_PADDING + (timestamp.getTime() - minTime) / timeRange * drawingWidth;
        ctx.fillText(timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), x, chartHeight - CHART_PADDING + 15);
    }

    // Draw price candles/lines
    ctx.strokeStyle = COLOR_BLACK;
    ctx.fillStyle = COLOR_BLACK;
    ctx.lineWidth = 1;

    for (let i = 0; i < chartData.prices.length - 1; i++) {
        const price1 = chartData.prices[i];
        const price2 = chartData.prices[i+1];
        
        const x1 = CHART_PADDING + (new Date(price1.t).getTime() - minTime) / timeRange * drawingWidth;
        const y1 = chartHeight - CHART_PADDING - (price1.p - minPrice) / priceRange * drawingHeight;
        
        const x2 = CHART_PADDING + (new Date(price2.t).getTime() - minTime) / timeRange * drawingWidth;
        const y2 = chartHeight - CHART_PADDING - (price2.p - minPrice) / priceRange * drawingHeight;

        // Draw lines between points
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Optionally draw vertical lines for candles if data included OHLC
        // For simplicity, drawing lines between closing prices.
    }

    // Draw technical indicators (moving averages, Bollinger Bands, etc.)
    if (chartData.technical_indicators) {
        // Draw SMA 20
        if (chartData.technical_indicators.sma_20 && Array.isArray(chartData.technical_indicators.sma_20)) {
            ctx.strokeStyle = '#0000FF'; // Blue for SMA 20
            ctx.lineWidth = 1.5;
            drawIndicatorLine(chartData.technical_indicators.sma_20, minTime, timeRange, drawingWidth, minPrice, priceRange, drawingHeight);
        }
        
        // Draw SMA 50
        if (chartData.technical_indicators.sma_50 && Array.isArray(chartData.technical_indicators.sma_50)) {
            ctx.strokeStyle = '#FF00FF'; // Magenta for SMA 50
            ctx.lineWidth = 1.5;
            drawIndicatorLine(chartData.technical_indicators.sma_50, minTime, timeRange, drawingWidth, minPrice, priceRange, drawingHeight);
        }
        
        // Draw Bollinger Bands
        if (chartData.technical_indicators.bb_upper && Array.isArray(chartData.technical_indicators.bb_upper) &&
            chartData.technical_indicators.bb_lower && Array.isArray(chartData.technical_indicators.bb_lower)) {
            // Upper band (red)
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 1;
            drawIndicatorLine(chartData.technical_indicators.bb_upper, minTime, timeRange, drawingWidth, minPrice, priceRange, drawingHeight);
            
            // Lower band (red)
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 1;
            drawIndicatorLine(chartData.technical_indicators.bb_lower, minTime, timeRange, drawingWidth, minPrice, priceRange, drawingHeight);
        }
    }

    // Draw Elliott Wave segments
    if (chartData.wave_levels) {
        chartData.wave_levels.forEach(level => {
            level.segments.forEach(segment => {
                const segmentStartTime = new Date(segment.start).getTime();
                const segmentEndTime = new Date(segment.end).getTime();
                
                // Determine color based on direction
                let waveColor = COLOR_BLACK; // Default or neutral
                if (segment.direction === 'up') {
                    waveColor = COLOR_WAVE_UP;
                } else if (segment.direction === 'down') {
                    waveColor = COLOR_WAVE_DOWN;
                }
                
                ctx.strokeStyle = waveColor;
                ctx.lineWidth = 1.5; // Slightly thicker for waves
                
                // Find the corresponding price points for segment start/end times
                // This requires careful mapping if segment times don't exactly match price data timestamps.
                // For now, assume approximation by nearest points.
                
                // Find the closest price data points for segment start and end
                const startPricePoint = chartData.prices.reduce((prev, curr) => 
                    Math.abs(new Date(curr.t).getTime() - segmentStartTime) < Math.abs(new Date(prev.t).getTime() - segmentStartTime) ? curr : prev
                );
                const endPricePoint = chartData.prices.reduce((prev, curr) => 
                    Math.abs(new Date(curr.t).getTime() - segmentEndTime) < Math.abs(new Date(prev.t).getTime() - segmentEndTime) ? curr : prev
                );

                if (startPricePoint && endPricePoint) {
                    const startX = CHART_PADDING + (new Date(startPricePoint.t).getTime() - minTime) / timeRange * drawingWidth;
                    const startY = chartHeight - CHART_PADDING - (startPricePoint.p - minPrice) / priceRange * drawingHeight;
                    
                    const endX = CHART_PADDING + (new Date(endPricePoint.t).getTime() - minTime) / timeRange * drawingWidth;
                    const endY = chartHeight - CHART_PADDING - (endPricePoint.p - minPrice) / priceRange * drawingHeight;

                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(endX, endY);
                    ctx.stroke();

                    // Draw labels or indicators if needed
                    ctx.fillStyle = waveColor;
                    ctx.font = '10px Arial';
                    ctx.fillText(segment.label || '', (startX + endX) / 2, (startY + endY) / 2 - 15); // Offset label above line
                }
            });
        });
    }
}

// Helper function to draw indicator lines
function drawIndicatorLine(indicatorValues, minTime, timeRange, drawingWidth, minPrice, priceRange, drawingHeight) {
    if (!chartData.prices || chartData.prices.length === 0) return;
    
    ctx.beginPath();
    
    // Align indicator values with price data points
    const startIndex = Math.max(0, chartData.prices.length - indicatorValues.length);
    
    for (let i = 0; i < indicatorValues.length && (startIndex + i) < chartData.prices.length; i++) {
        const pricePoint = chartData.prices[startIndex + i];
        if (pricePoint && indicatorValues[i] !== null && !isNaN(indicatorValues[i])) {
            const x = CHART_PADDING + (new Date(pricePoint.t).getTime() - minTime) / timeRange * drawingWidth;
            const y = chartHeight - CHART_PADDING - (indicatorValues[i] - minPrice) / priceRange * drawingHeight;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
    }
    
    ctx.stroke();
}

// --- Event Handlers ---
setSymbolBtn.addEventListener('click', async () => {
    const symbol = symbolInput.value.trim().toUpperCase();
    const interval = intervalSelect.value;
    const source = sourceSelect.value;

    if (!symbol) {
        alert('Please enter a symbol.');
        return;
    }

    console.log(`Setting symbol: ${symbol}, Interval: ${interval}, Source: ${source}`);

    try {
        const response = await fetch(`${API_BASE_URL}/symbol`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbol: symbol, interval: interval, source_preference: source || undefined }), // source can be empty
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Symbol set successfully:', result.message);
            // Clear previous chart data to avoid confusion while new data loads
            chartData.prices = [];
            chartData.wave_levels = [];
            chartData.candidates = [];
            chartData.symbol = symbol;
            chartData.timestamp = '';
            chartData.current_price = null;
            renderChart(); // Clear the chart
            updateUI({ symbol: symbol, current_price: null, timestamp: '' }); // Clear UI
        } else {
            const errorData = await response.json();
            console.error('Error setting symbol:', response.status, errorData.detail);
            alert(`Error setting symbol: ${errorData.detail}`);
        }
    } catch (error) {
        console.error('Network error:', error);
        alert('Failed to connect to the backend. Please check the server status.');
    }
});

// --- Initialization ---
function initChart() {
    ctx = priceChartCanvas.getContext('2d');
    if (!ctx) {
        console.error("Could not get 2D context for canvas.");
        return;
    }
    chartWidth = priceChartCanvas.clientWidth;
    chartHeight = priceChartCanvas.clientHeight;
    
    // Set initial canvas dimensions to match CSS
    priceChartCanvas.width = chartWidth;
    priceChartCanvas.height = chartHeight;

    // Add event listener for window resize to re-render chart appropriately
    window.addEventListener('resize', () => {
        chartWidth = priceChartCanvas.clientWidth;
        chartHeight = priceChartCanvas.clientHeight;
        priceChartCanvas.width = chartWidth; // Update canvas resolution
        priceChartCanvas.height = chartHeight;
        renderChart(); // Re-render chart on resize
    });

    console.log(`Canvas initialized: ${chartWidth}x${chartHeight}`);
}

// Main initialization function
async function initApp() {
    initChart();
    connectWebSocket();
    // Initial UI update in case no data is loaded yet
    updateUI({ symbol: 'N/A', current_price: null, timestamp: '' });
    confidenceScoresDiv.innerHTML = '<p>Enter a symbol and click "Set Symbol" to begin.</p>';
    waveCandidatesDiv.innerHTML = '<p>Analysis results will appear here.</p>';
}

// Run initialization when the DOM is ready
document.addEventListener('DOMContentLoaded', initApp);