"""Configuration settings for the Swing Trading System."""

from pathlib import Path
from typing import List

# Project paths
# Works on both local and Streamlit Cloud
# Note: On Streamlit Cloud, files are ephemeral (reset on restart), which is fine for caching
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Database
# SQLite works on Streamlit Cloud (ephemeral storage - data resets on restart)
# This is acceptable for caching - app will fetch fresh data when needed
DB_PATH = DATA_DIR / "trading_data.db"

# Market indices for regime filter
NIFTY_50_SYMBOL = "^NSEI"
SP500_SYMBOL = "^GSPC"
MARKET_INDEX = NIFTY_50_SYMBOL  # Default to Nifty 50

# Trading parameters
RISK_PER_TRADE = 0.01  # 1% of equity per trade
ACCOUNT_EQUITY = 100000  # Account equity for position sizing
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0  # For stop loss calculation (Low - 2 * ATR)
RISK_REWARD_RATIO = 2.0  # Target is 2x the risk (1:2 ratio)
MIN_LOOKBACK_DAYS = 250  # Minimum data required for 200 EMA

# Calibration parameters
MIN_OPPORTUNITIES_THRESHOLD = 5  # Auto-calibrate if fewer than 5 opportunities
CALIBRATION_STEP = 0.005  # 0.5% increment per calibration attempt
MAX_CALIBRATION_ATTEMPTS = 3

# Dead Stock Filter parameters
MIN_ADX_THRESHOLD = 20.0  # Minimum ADX to avoid sideways markets
MIN_VOLUME_THRESHOLD = 100_000  # Minimum average volume (100k shares)

# Cognition Engine parameters
VIX_PANIC_THRESHOLD = 22.0  # VIX > 22 = Panic (halve positions)
VIX_CALM_THRESHOLD = 18.0  # VIX < 18 = Calm (normal sizing)

# Swing Trading Parameters
SWING_HOLDING_PERIOD_MIN = 10  # Minimum holding period (days)
SWING_HOLDING_PERIOD_MAX = 20  # Maximum holding period (days)
SWING_STOP_PERCENT = 0.02  # 2% below support level
SWING_TARGET_RISK_MULTIPLIER = 2.0  # Target 2: 2.0x Risk

# Strategy parameters (swing trading strategies)
INSTITUTIONAL_WAVE_EMA_50 = 50  # 50 EMA period
INSTITUTIONAL_WAVE_EMA_200 = 200  # 200 EMA period
GOLDEN_POCKET_FIB_LEVEL = 0.618  # Fibonacci golden pocket level
COIL_BREAKOUT_RANGE_THRESHOLD = 0.05  # 5% price range for base
COIL_BREAKOUT_VOLUME_MULTIPLIER = 1.5  # Volume multiplier for breakout
DARVAS_BOX_ADX_MIN = 25.0  # Minimum ADX for Darvas Box
WEEKLY_CLIMBER_RSI_MIN = 50.0  # Minimum RSI for Weekly Climber
WEEKLY_CLIMBER_RSI_MAX = 65.0  # Maximum RSI for Weekly Climber
MACD_ZERO_TURN_HISTOGRAM_DAYS = 5  # Days MACD histogram should be red

# Calibration parameters
CALIBRATION_MIN_OPPORTUNITIES = 5  # Minimum opportunities before relaxing
CALIBRATION_SNIPER_ADX = 30  # Sniper mode ADX
CALIBRATION_RIFLE_ADX = 25  # Rifle mode ADX
CALIBRATION_SHOTGUN_ADX = 20  # Shotgun mode ADX

# Strategy Configuration (Plug-and-Play - Swing Trading)
STRATEGY_CONFIG = {
    "institutional_wave": {
        "enabled": True,
        "params": {
            "ema_50_period": INSTITUTIONAL_WAVE_EMA_50,
            "ema_200_period": INSTITUTIONAL_WAVE_EMA_200,
        },
        "allowed_regimes": ["AGGRESSIVE", "DEFENSIVE"],
        "group": "A",  # Support
    },
    "golden_pocket": {
        "enabled": True,
        "params": {
            "fib_level": GOLDEN_POCKET_FIB_LEVEL,
        },
        "allowed_regimes": ["AGGRESSIVE", "DEFENSIVE"],
        "group": "A",  # Support
    },
    "coil_breakout": {
        "enabled": True,
        "params": {
            "range_threshold": COIL_BREAKOUT_RANGE_THRESHOLD,
            "volume_multiplier": COIL_BREAKOUT_VOLUME_MULTIPLIER,
        },
        "allowed_regimes": ["AGGRESSIVE"],
        "group": "B",  # Breakout
    },
    "darvas_box": {
        "enabled": True,
        "params": {
            "min_adx": DARVAS_BOX_ADX_MIN,
        },
        "allowed_regimes": ["AGGRESSIVE"],
        "group": "B",  # Breakout
    },
    "weekly_climber": {
        "enabled": True,
        "params": {
            "rsi_min": WEEKLY_CLIMBER_RSI_MIN,
            "rsi_max": WEEKLY_CLIMBER_RSI_MAX,
        },
        "allowed_regimes": ["AGGRESSIVE", "DEFENSIVE"],
        "group": "C",  # Momentum
    },
    "macd_zero_turn": {
        "enabled": True,
        "params": {
            "histogram_days": MACD_ZERO_TURN_HISTOGRAM_DAYS,
        },
        "allowed_regimes": ["AGGRESSIVE", "DEFENSIVE"],
        "group": "C",  # Momentum
    },
}

# Strategy Directory for custom strategies (future use)
STRATEGY_DIRECTORY = PROJECT_ROOT / "strategies"

# Backtest parameters
BACKTEST_LOOKBACK_YEARS = 2
INITIAL_CAPITAL = 100000  # Starting capital for backtests

# Performance settings
MAX_WORKERS = 20  # ThreadPoolExecutor workers for parallel fetching
CACHE_EXPIRY_HOURS = 24  # Re-download if data older than 24 hours

# Nifty 500 tickers (Top 150 for demo - can be extended to full 500)
# In production, load from CSV or NSE API
NIFTY_500_TICKERS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "BAJFINANCE.NS", "LICI.NS",
    "ITC.NS", "HCLTECH.NS", "AXISBANK.NS", "KOTAKBANK.NS", "LT.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
    "WIPRO.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "NESTLEIND.NS",
    "COALINDIA.NS", "TATAMOTORS.NS", "DIVISLAB.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "GRASIM.NS",
    "ULTRACEMCO.NS", "SHREECEM.NS", "DABUR.NS", "BRITANNIA.NS", "GODREJCP.NS",
    "PIDILITIND.NS", "BAJAJFINSV.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    "M&M.NS", "MOTHERSON.NS", "BOSCHLTD.NS", "BALKRISIND.NS", "MRF.NS",
    "APOLLOHOSP.NS", "DRREDDY.NS", "CIPLA.NS", "TORNTPHARM.NS", "LUPIN.NS",
    "AUROPHARMA.NS", "GLENMARK.NS", "CADILAHC.NS", "BIOCON.NS", "LALPATHLAB.NS",
    "ZOMATO.NS", "SWIGGY.NS", "PAYTM.NS", "ZYDUSLIFE.NS", "ALKEM.NS",
    "INDIGO.NS", "INTERGLOBE.NS", "INDIGOPNTS.NS", "JINDALSAW.NS", "JSWENERGY.NS",
    "ADANIGREEN.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "RPOWER.NS", "NHPC.NS",
    "SJVN.NS", "IRCTC.NS", "CONCOR.NS", "CONCOR.NS", "CONCOR.NS",
    "DMART.NS", "TRENT.NS", "SHOPPERSSTOP.NS", "VBL.NS", "RADICO.NS",
    "UNIONBANK.NS", "PNB.NS", "CANBK.NS", "BANKBARODA.NS", "INDIANB.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "RBLBANK.NS", "YESBANK.NS", "INDUSINDBK.NS",
    "BANDHANBNK.NS", "SOUTHBANK.NS", "UCOBANK.NS", "CENTRALBK.NS", "ORIENTBANK.NS",
    "TECHM.NS", "MINDTREE.NS", "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS",
    "COFORGE.NS", "ZENSAR.NS", "SONATA.NS", "NEWGEN.NS", "INTELLECT.NS",
    "TATAELXSI.NS", "LTTS.NS", "CYIENT.NS", "KPITTECH.NS", "L&TINFRA.NS",
    "IRB.NS", "ADANIPORTS.NS", "JKCEMENT.NS", "RAMCOCEM.NS", "ACC.NS",
    "AMBUJACEM.NS", "HEIDELBERG.NS", "ORIENTCEM.NS", "PRISM.NS", "BIRLACORPN.NS",
    "GODREJPROP.NS", "DLF.NS", "PRESTIGE.NS", "SOBHA.NS", "BRIGADE.NS",
    "OBEROIRLTY.NS", "PHOENIXLTD.NS", "INDIABULLS.NS", "MAGMA.NS", "KOLTEPATIL.NS",
    "MAHINDRA.NS", "ESCORTS.NS", "ASHOKLEY.NS", "EICHERMOT.NS", "TVSMOTOR.NS",
    "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "FORCEMOT.NS", "ATUL.NS", "SRF.NS",
    "UPL.NS", "RALLIS.NS", "DEEPAKNTR.NS", "GHCL.NS", "TATACHEM.NS",
    "GUJALKALI.NS", "DCMSHRIRAM.NS", "CHAMBLFERT.NS", "COROMANDEL.NS", "NAGARFERT.NS",
    "RCF.NS", "GSFC.NS", "MFL.NS", "FACT.NS", "SPIC.NS",
    "GNFC.NS", "RASHTRIYACEM.NS", "JKLAKSHMI.NS", "KCP.NS", "SHRIRAMFIN.NS",
    "BAJAJFINSV.NS", "M&MFIN.NS", "CHOLAFIN.NS", "SHRIRAMCIT.NS", "LICHSGFIN.NS",
    "MUTHOOTFIN.NS", "MANAPPURAM.NS", "M&MFIN.NS", "CHOLAFIN.NS", "SHRIRAMCIT.NS",
]

# Function to load tickers from CSV (if available)
def load_nifty_500_from_csv(csv_path: str | None = None) -> List[str]:
    """
    Load Nifty 500 tickers from CSV file.
    
    Expected CSV format: ticker column with values like "RELIANCE.NS"
    """
    if csv_path is None:
        csv_path = DATA_DIR / "nifty_500_tickers.csv"
    
    if Path(csv_path).exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "ticker" in df.columns:
            return df["ticker"].tolist()
    
    return NIFTY_500_TICKERS

