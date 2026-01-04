# Medium-Term Swing Trading Dashboard - Nifty 500

Production-grade Swing Trading System with Material Design Dashboard for Nifty 500 analysis.

## Features

- **6 Swing Trading Strategies**: 
  - The Institutional Wave (50 EMA)
  - The Golden Pocket (Fibonacci)
  - The Coil Breakout (Base Building)
  - The Darvas Box (Trend Ladder)
  - The Weekly Climber (Multi-Timeframe)
  - The MACD Zero-Turn (Momentum Shift)
- **Market Intelligence**: Real-time regime diagnosis (AGGRESSIVE/DEFENSIVE/CASH_PROTECTION), VIX analysis, sector heatmap
- **Smart Caching**: Uses SQLite database (`trading_data.db`) by default, with Hard Refresh option for fresh data
- **Risk Management**: Strategy-group-specific stop losses and dual targets (Swing High + 2x Risk)
- **Quality Scoring**: Swing Score (0-100) based on Pattern Clarity, Sector Strength, and Risk:Reward
- **Material Design UI**: Modern, light-themed dashboard with interactive charts
- **Parallel Data Fetching**: Fast batch data loading with rate limiting

## Setup

### Local Development

#### 1. Initialize Project with uv

```bash
uv init
uv add pandas pandas-ta pydantic plotly streamlit yfinance numpy requests
```

#### 2. Run the Dashboard

```bash
streamlit run app.py
```

### Deploy to Streamlit Cloud (Free)

This app is fully configured for deployment to Streamlit Cloud and is mobile-accessible.

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

#### Deployment Steps

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set the main file path to: `app.py`
   - Click "Deploy"

3. **Access from Mobile**
   - Once deployed, you'll get a public URL (e.g., `https://your-app.streamlit.app`)
   - The app is fully responsive and optimized for mobile devices
   - Open the URL on any mobile browser

#### Important Notes for Streamlit Cloud

- **Database**: SQLite database works on Streamlit Cloud but is ephemeral (resets on restart). This is fine for caching - the app will fetch fresh data when needed.
- **No Configuration Required**: The app is configured to work out-of-the-box on Streamlit Cloud.
- **Free Tier**: Streamlit Cloud free tier is sufficient for this app.
- **Mobile Optimized**: The UI automatically adapts to mobile screens with responsive CSS.

## Project Structure

```
.
├── app.py                          # Main Streamlit dashboard
├── config.py                       # Configuration settings
├── pyproject.toml                  # Dependencies
├── data/                           # Data storage
│   ├── trading_data.db            # SQLite cache for OHLCV data
│   ├── company_cache.json         # Company metadata cache
│   └── nifty500_tickers_cache.csv # Nifty 500 ticker list
└── src/
    ├── engine/                     # Trading logic
    │   ├── strategies.py          # StrategyRunner
    │   ├── strategy_modules/      # 6 swing trading strategies
    │   ├── strategy_base.py       # Base strategy class
    │   ├── strategy_registry.py   # Strategy registry
    │   ├── analyst.py             # Intelligence engine (thesis, scoring)
    │   ├── assistant.py           # Trade assistant (position sizing)
    │   ├── calibration.py         # Quality tagging (HIGH/STANDARD/LOOSE)
    │   ├── cognition.py           # Market intelligence
    │   ├── risk.py                # Swing risk engine
    │   └── quant_math.py          # Quantitative calculations
    ├── feed/                       # Data ingestion
    │   ├── data.py                # FastDataEngine (parallel fetching, caching)
    │   └── enrichment.py          # Company enrichment (sector mapping)
    └── models/                     # Data models
        └── schemas.py             # Pydantic schemas (TradeTicket, TradeSignal)
```

## Usage

1. **Market Scan**: Click "🔍 Scan Nifty 500" to find current opportunities
2. **Hard Refresh**: Click "🔄 Hard Refresh Data" to fetch fresh data from yfinance (default uses cache)
3. **View Opportunities**: Browse filtered and sorted opportunities in the left panel
4. **Inspect Details**: Click any opportunity to view trade details, analysis, and strategy-specific charts
5. **Market Intelligence**: View real-time market regime, VIX, sector performance, and health score

## Key Components

- **FastDataEngine**: Parallel data fetching with SQLite caching and rate limiting
- **StrategyRunner**: Plug-and-play strategy execution with regime filtering
- **IntelligenceEngine**: Generates investment theses and calculates swing scores
- **CalibrationEngine**: Tags trades with quality (HIGH/STANDARD/LOOSE) based on characteristics
- **CognitionEngine**: Diagnoses market regime, volatility, and sector performance

## Configuration

Key settings in `config.py`:
- `ACCOUNT_EQUITY`: Account equity for position sizing
- `MAX_WORKERS`: ThreadPoolExecutor workers for parallel fetching
- `CACHE_EXPIRY_HOURS`: Cache validity period (default: 24 hours)
- `STRATEGY_CONFIG`: Strategy parameters and regime settings

## Notes

- Data is cached in `trading_data.db` by default for fast scans
- Use Hard Refresh to fetch fresh data from yfinance
- Rate limiting (0.15s delay) prevents IP blocking
- Weekend guardrails ensure valid trading day data
