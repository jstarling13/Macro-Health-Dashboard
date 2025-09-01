import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import time
import logging
from typing import Dict, Tuple, Optional, Any
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed - USD data will use fallback")

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False
    logger.warning("streamlit_autorefresh not installed - auto-refresh disabled")

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="Market Health Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modular scoring configuration
SCORING_CONFIG = {
    'gdp': {
        'name': 'GDP Growth',
        'weight': 1.0,
        'excellent': {'min_latest': 3.0, 'min_avg': 2.5, 'score': 20},
        'good': {'min_latest': 2.0, 'min_avg': 2.0, 'score': 16},
        'fair': {'min_latest': 1.0, 'min_avg': 0, 'score': 12},
        'poor': {'min_latest': 0, 'min_avg': 0, 'score': 8},
        'critical': {'score': 4}
    },
    'fed': {
        'name': 'Fed Policy',
        'weight': 1.0,
        'ranges': {
            20: {'max_rate': 2.0},
            16: {'max_rate': 3.5},
            12: {'max_rate': 5.0},
            8: {'max_rate': 6.0},
            4: {'max_rate': float('inf')}
        },
        'sentiment_adjustment': {'Dovish': 2, 'Hawkish': -2, 'Neutral': 0}
    },
    'earnings': {
        'name': 'Earnings',
        'weight': 1.0,
        'excellent': {'min_beat': 75, 'min_avg': 72, 'score': 20},
        'good': {'min_beat': 70, 'score': 16},
        'fair': {'min_beat': 65, 'score': 12},
        'poor': {'min_beat': 60, 'score': 8},
        'critical': {'score': 4},
        'baseline_beat_rate': 72
    },
    'inflation': {
        'name': 'Inflation',
        'weight': 1.0,
        'excellent': {'max_deviation': 0.5, 'max_core_deviation': 0.5, 'score': 20},
        'good': {'max_deviation': 1.0, 'trend': 'Falling', 'score': 16},
        'fair': {'max_deviation': 1.5, 'score': 12},
        'poor': {'max_deviation': 2.5, 'score': 8},
        'critical': {'score': 4}
    },
    'usd': {
        'name': 'USD Strength',
        'weight': 1.0,
        'excellent': {'min_dxy': 103, 'max_dxy': 107, 'trend_boost': True, 'score': 20},
        'good': {'min_dxy': 100, 'max_dxy': 110, 'score': 16},
        'fair': {'min_dxy': 95, 'max_dxy': 115, 'score': 12},
        'poor': {'min_dxy': 90, 'max_dxy': 120, 'score': 8},
        'critical': {'score': 4}
    },
    'overall_grades': {
        90: {'grade': 'A', 'color': '#10b981', 'sentiment': 'positive'},
        80: {'grade': 'B', 'color': '#3b82f6', 'sentiment': 'positive'},
        70: {'grade': 'C', 'color': '#f59e0b', 'sentiment': 'neutral'},
        0: {'grade': 'F', 'color': '#ef4444', 'sentiment': 'negative'}
    }
}

COLOR_PALETTE = {
    'gdp': '#1f77b4',
    'fed': '#ff7f0e', 
    'earnings': '#2ca02c',
    'inflation': '#d62728',
    'usd': '#9467bd'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_api_key(key_name: str) -> Optional[str]:
    """Get API key from hardcoded values, Streamlit secrets, or environment"""
    
    # Hardcoded API keys for convenience
    HARDCODED_KEYS = {
        "FRED_API_KEY": "6d9f556dbe4a84d6bf65a4833caa4aa9",
        "ALPHA_VANTAGE_API_KEY": "0WBEM03QUNESN41X"
    }
    
    # First try hardcoded keys
    if key_name in HARDCODED_KEYS:
        return HARDCODED_KEYS[key_name]
    
    try:
        # Then try Streamlit secrets
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception as e:
        logger.debug(f"Could not access Streamlit secrets: {str(e)}")
    
    try:
        # Fall back to environment variables
        return os.getenv(key_name)
    except Exception as e:
        logger.debug(f"Could not access environment variable {key_name}: {str(e)}")
    
    return None

def safe_api_call(url: str, params: dict, timeout: int = 10) -> Optional[dict]:
    """Make safe API call with proper error handling"""
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"API call timeout: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {url} - {str(e)}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in API call: {str(e)}")
        return None

def load_css() -> str:
    """Load CSS styling"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .banner-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .kpi-card {
        background: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    
    .kpi-title {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .kpi-subtitle {
        font-size: 0.85rem;
        opacity: 0.7;
        font-weight: 400;
    }
    
    .grade-gauge-container {
        background: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .data-warning {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #92400e;
    }
    
    .error-message {
        background-color: #fee2e2;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #dc2626;
    }
    </style>
    """

# ============================================================================
# CACHED DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_data_cached(series_id: str, start_date: str, fred_api_key: Optional[str], **params) -> Optional[pd.DataFrame]:
    """Generic FRED data fetcher - cached standalone function"""
    if not fred_api_key:
        logger.warning("FRED API key not found")
        return None
        
    url = "https://api.stlouisfed.org/fred/series/observations"
    params.update({
        "series_id": series_id,
        "api_key": fred_api_key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "desc"
    })
    
    data = safe_api_call(url, params)
    if not data or "observations" not in data:
        return None
        
    try:
        df = pd.DataFrame(data["observations"])
        if df.empty:
            return None
            
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        # Remove rows with invalid dates or values
        df = df.dropna(subset=['date', 'value'])
        
        if df.empty:
            return None
            
        return df.sort_values('date')
    except Exception as e:
        logger.error(f"Error processing FRED data for {series_id}: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_data_cached(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch yfinance data - cached standalone function"""
    if not HAS_YFINANCE:
        return None
        
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=period, interval="1d")
        
        if hist.empty:
            return None
            
        return hist.reset_index()
    except Exception as e:
        logger.error(f"yfinance data fetch failed for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_multiple_usd_sources(fred_api_key: Optional[str]) -> Optional[pd.DataFrame]:
    """Fetch USD data from multiple sources with fallback chain"""
    
    # Source 1: yfinance DXY (most reliable)
    if HAS_YFINANCE:
        for attempt in range(3):
            try:
                # Prioritize the actual DXY ticker first
                tickers_to_try = ["DX-Y.NYB", "UUP"]
                ticker = tickers_to_try[attempt] if attempt < len(tickers_to_try) else "DX-Y.NYB"
                
                time.sleep(attempt * 1)  # Reduced delay
                hist_data = fetch_yfinance_data_cached(ticker, "2y")
                
                if hist_data is not None and not hist_data.empty:
                    if 'Date' in hist_data.columns:
                        usd_data = pd.DataFrame({
                            'date': pd.to_datetime(hist_data['Date']),
                            'dxy_close': hist_data['Close'].values
                        }).dropna().reset_index(drop=True)
                    else:
                        usd_data = pd.DataFrame({
                            'date': pd.to_datetime(hist_data.index),
                            'dxy_close': hist_data['Close'].values
                        }).dropna().reset_index(drop=True)

                    if not usd_data.empty:
                        # Only apply scaling for UUP ETF, keep DXY as-is
                        if ticker == "UUP":
                            # UUP is roughly 1/4 the value of DXY, so scale up
                            usd_data['dxy_close'] = usd_data['dxy_close'] * 4.0
                        
                        usd_monthly = usd_data.set_index('date').resample('M').last().reset_index()
                        usd_monthly['month_year'] = usd_monthly['date'].dt.strftime('%b %Y')
                        logger.info(f"Successfully fetched USD data from yfinance ({ticker})")
                        return usd_monthly.tail(24)
                        
            except Exception as e:
                logger.warning(f"yfinance attempt {attempt+1} failed: {str(e)}")
                if "rate limit" in str(e).lower():
                    time.sleep(3 * (attempt + 1))
                continue
    
    # Source 2: FRED USD/EUR (fallback only)
    if fred_api_key:
        try:
            start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
            df = fetch_fred_data_cached("DEXUSEU", start_date, fred_api_key)
            if df is not None and not df.empty:
                df = df.rename(columns={'value': 'usd_eur_rate'})
                # Simple inverse relationship approximation
                df['dxy_close'] = 108 / df['usd_eur_rate']  # Adjusted for current levels
                df = df[['date', 'dxy_close']].dropna()
                
                if not df.empty:
                    usd_monthly = df.set_index('date').resample('M').last().reset_index()
                    usd_monthly['month_year'] = usd_monthly['date'].dt.strftime('%b %Y')
                    logger.info("Successfully fetched USD data from FRED (USD/EUR) as fallback")
                    return usd_monthly.tail(24)
        except Exception as e:
            logger.warning(f"FRED USD data failed: {str(e)}")
    
    return None

# ============================================================================
# FALLBACK DATA GENERATORS
# ============================================================================

def generate_gdp_fallback_data() -> pd.DataFrame:
    """Generate realistic GDP fallback data"""
    try:
        base_date = datetime(2022, 1, 1)
        dates = [base_date + timedelta(days=90*i) for i in range(12)]
        growth_rates = [2.1, 1.8, 2.9, 2.6, 4.2, 2.1, 3.2, 4.9, 2.4, 2.8, 2.5, 2.7]
        quarters = [f"Q{((d.month-1)//3)+1} {d.year}" for d in dates]
        
        return pd.DataFrame({
            'date': pd.to_datetime(dates),
            'gdp_growth': growth_rates,
            'quarter': quarters
        })
    except Exception as e:
        logger.error(f"Error generating GDP fallback data: {str(e)}")
        return pd.DataFrame()

def generate_fed_fallback_data() -> Tuple[pd.DataFrame, float, str]:
    """Generate realistic Fed rate fallback data"""
    try:
        base_date = datetime(2020, 1, 1)
        dates = [base_date + timedelta(days=30*i) for i in range(60)]
        rates = ([1.75, 1.0, 0.25] + [0.25]*15 + 
                [0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.25, 5.5] + 
                [5.5]*4 + [5.25, 5.0, 4.75, 4.5, 4.25, 4.0, 3.75] + [3.5]*15)
        
        min_len = min(len(dates), len(rates))
        dates = dates[:min_len]
        rates = rates[:min_len]
        month_years = [d.strftime('%b %Y') for d in dates]
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'fed_rate': rates,
            'month_year': month_years
        })
        
        current_rate = rates[-1] if rates else 3.5
        return df, current_rate, "Neutral"
    except Exception as e:
        logger.error(f"Error generating Fed fallback data: {str(e)}")
        return pd.DataFrame(), 3.5, "Neutral"

def generate_earnings_fallback_data() -> pd.DataFrame:
    """Generate realistic earnings fallback data"""
    try:
        quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023", 
                   "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
        beat_rates = [68.5, 72.1, 75.3, 69.8, 71.2, 73.7, 70.4, 68.9]
        miss_rates = [100 - rate for rate in beat_rates]
        
        return pd.DataFrame({
            'quarter': quarters,
            'beat_rate': beat_rates,
            'miss_rate': miss_rates,
            'total_companies': [500] * len(quarters),
            'companies_beat': [int(rate * 5) for rate in beat_rates],
            'companies_missed': [int((100-rate) * 5) for rate in beat_rates]
        })
    except Exception as e:
        logger.error(f"Error generating earnings fallback data: {str(e)}")
        return pd.DataFrame()

def generate_inflation_fallback_data() -> pd.DataFrame:
    """Generate realistic inflation fallback data"""
    try:
        base_date = datetime(2022, 1, 1)
        dates = [base_date + timedelta(days=30*i) for i in range(24)]
        cpi_rates = [7.5, 7.9, 8.5, 8.3, 8.6, 9.1, 8.5, 8.2, 7.7, 7.1, 6.5, 6.4,
                    6.0, 5.4, 4.9, 4.0, 3.7, 3.2, 3.1, 3.0, 3.4, 3.2, 2.9, 2.4]
        core_rates = [6.0, 6.2, 6.5, 6.2, 6.0, 5.9, 5.7, 5.9, 6.1, 5.8, 5.5, 5.7,
                     5.5, 5.2, 4.9, 4.4, 4.1, 3.8, 3.6, 3.3, 3.2, 3.0, 2.8, 2.6]
        
        min_len = min(len(dates), len(cpi_rates), len(core_rates))
        dates = dates[:min_len]
        cpi_rates = cpi_rates[:min_len]
        core_rates = core_rates[:min_len]
        month_years = [d.strftime('%b %Y') for d in dates]
        
        return pd.DataFrame({
            'date': pd.to_datetime(dates),
            'cpi_yoy': cpi_rates,
            'core_cpi_yoy': core_rates,
            'month_year': month_years
        })
    except Exception as e:
        logger.error(f"Error generating inflation fallback data: {str(e)}")
        return pd.DataFrame()

def generate_usd_fallback_data() -> pd.DataFrame:
    """Generate realistic USD fallback data"""
    try:
        base_date = datetime(2023, 1, 1)
        dates = [base_date + timedelta(days=30*i) for i in range(24)]
        # Updated to reflect current DXY levels around 97
        dxy_values = [101.2, 100.8, 100.1, 99.9, 100.4, 100.8, 101.1, 101.6, 
                     101.2, 100.9, 100.3, 99.8, 99.2, 98.9, 99.1, 100.1, 
                     100.8, 101.4, 99.9, 98.2, 97.8, 97.5, 97.1, 97.3]
        
        min_len = min(len(dates), len(dxy_values))
        dates = dates[:min_len]
        dxy_values = dxy_values[:min_len]
        month_years = [d.strftime('%b %Y') for d in dates]
        
        return pd.DataFrame({
            'date': pd.to_datetime(dates),
            'dxy_close': dxy_values,
            'month_year': month_years
        })
    except Exception as e:
        logger.error(f"Error generating USD fallback data: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# MARKET HEALTH ANALYZER CLASS
# ============================================================================

class MarketHealthAnalyzer:
    """Market health analysis and scoring system"""
    
    def __init__(self):
        self.fred_api_key = get_api_key("FRED_API_KEY")
        self.fallback_status = {
            'gdp': False,
            'fed': False,
            'earnings': False,
            'inflation': False,
            'usd': False
        }
        
    def fetch_gdp_data(self) -> pd.DataFrame:
        """Fetch Real GDP growth data with robust error handling"""
        try:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            df = fetch_fred_data_cached("A191RL1Q225SBEA", start_date, self.fred_api_key)
            
            if df is not None and not df.empty:
                df = df.rename(columns={'value': 'gdp_growth'})
                df["quarter"] = df["date"].dt.to_period("Q").astype(str)
                return df.tail(12)
            
        except Exception as e:
            logger.error(f"Error fetching GDP data: {str(e)}")
        
        logger.info("Using fallback GDP data")
        self.fallback_status['gdp'] = True
        return generate_gdp_fallback_data()

    def fetch_fed_data(self) -> Tuple[pd.DataFrame, float, str]:
        """Fetch Federal Reserve rate data with robust error handling"""
        try:
            start_date = (datetime.now() - timedelta(days=7*365)).strftime('%Y-%m-%d')
            df = fetch_fred_data_cached("FEDFUNDS", start_date, self.fred_api_key)
            
            if df is not None and not df.empty:
                df = df.rename(columns={'value': 'fed_rate'}).tail(60)
                df["month_year"] = df["date"].dt.strftime("%b %Y")
                
                current_rate = float(df["fed_rate"].iloc[-1])
                
                if len(df) >= 6:
                    recent_rates = df["fed_rate"].tail(6)
                    rate_change = recent_rates.iloc[-1] - recent_rates.iloc[0]
                    
                    if rate_change < -0.25:
                        sentiment = "Dovish"
                    elif rate_change > 0.25:
                        sentiment = "Hawkish"
                    else:
                        sentiment = "Neutral"
                else:
                    sentiment = "Neutral"
                    
                return df, current_rate, sentiment
                
        except Exception as e:
            logger.error(f"Error fetching Fed data: {str(e)}")
        
        logger.info("Using fallback Fed data")
        self.fallback_status['fed'] = True
        return generate_fed_fallback_data()

    def fetch_earnings_data(self) -> pd.DataFrame:
        """Generate earnings data - using fallback by default"""
        logger.info("Using fallback earnings data")
        self.fallback_status['earnings'] = True
        return generate_earnings_fallback_data()

    def fetch_inflation_data(self) -> pd.DataFrame:
        """Fetch CPI and Core CPI data with robust error handling"""
        try:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            cpi_df = fetch_fred_data_cached("CPIAUCSL", start_date, self.fred_api_key, units="pc1")
            core_df = fetch_fred_data_cached("CPILFESL", start_date, self.fred_api_key, units="pc1")

            if (cpi_df is not None and not cpi_df.empty and 
                core_df is not None and not core_df.empty):
                
                cpi_df = cpi_df.rename(columns={'value': 'cpi_yoy'})
                core_df = core_df.rename(columns={'value': 'core_cpi_yoy'})
                
                inflation_df = pd.merge(
                    cpi_df[['date', 'cpi_yoy']],
                    core_df[['date', 'core_cpi_yoy']],
                    on='date', how='inner'
                )
                
                if not inflation_df.empty:
                    inflation_df["month_year"] = inflation_df["date"].dt.strftime("%b %Y")
                    return inflation_df.tail(24)

        except Exception as e:
            logger.error(f"Error fetching inflation data: {str(e)}")

        logger.info("Using fallback inflation data")
        self.fallback_status['inflation'] = True
        return generate_inflation_fallback_data()

    def fetch_usd_data(self) -> pd.DataFrame:
        """Fetch USD Index data with multiple reliable sources"""
        try:
            usd_data = fetch_multiple_usd_sources(self.fred_api_key)
            if usd_data is not None and not usd_data.empty:
                return usd_data
        except Exception as e:
            logger.error(f"Error fetching USD data: {str(e)}")

        logger.info("Using fallback USD data")
        self.fallback_status['usd'] = True
        return generate_usd_fallback_data()

    def calculate_gdp_score(self, gdp_df: pd.DataFrame) -> Tuple[int, str, dict]:
        """Calculate GDP score with error handling"""
        try:
            if gdp_df.empty or 'gdp_growth' not in gdp_df.columns:
                return 12, "No GDP data available", {}
            
            latest_growth = float(gdp_df['gdp_growth'].iloc[-1])
            avg_growth = float(gdp_df['gdp_growth'].tail(4).mean())
            trend = "Rising" if len(gdp_df) > 1 and latest_growth > gdp_df['gdp_growth'].iloc[-2] else "Falling"
            
            metadata = {
                'latest_growth': latest_growth,
                'avg_growth': avg_growth,
                'trend': trend,
                'latest_quarter': gdp_df.get('quarter', ['N/A']).iloc[-1]
            }
            
            config = SCORING_CONFIG['gdp']
            
            if latest_growth >= config['excellent']['min_latest'] and avg_growth >= config['excellent']['min_avg']:
                return config['excellent']['score'], "Strong GDP growth above trend", metadata
            elif latest_growth >= config['good']['min_latest'] and avg_growth >= config['good']['min_avg']:
                return config['good']['score'], "Solid GDP growth at trend", metadata
            elif latest_growth >= config['fair']['min_latest']:
                return config['fair']['score'], "Moderate GDP growth below trend", metadata
            elif latest_growth >= config['poor']['min_latest']:
                return config['poor']['score'], "Weak but positive GDP growth", metadata
            else:
                return config['critical']['score'], "GDP contraction - recession risk", metadata
                
        except Exception as e:
            logger.error(f"Error calculating GDP score: {str(e)}")
            return 12, "Error calculating GDP score", {}

    def calculate_fed_score(self, current_rate: float, sentiment: str) -> Tuple[int, str, dict]:
        """Calculate Fed score with error handling"""
        try:
            if current_rate is None:
                return 12, "No Fed data available", {}
                
            metadata = {
                'current_rate': float(current_rate),
                'sentiment': sentiment,
                'policy_stance': ('Restrictive' if current_rate > 4.5 else 
                                'Neutral' if current_rate > 2.5 else 'Accommodative')
            }
            
            config = SCORING_CONFIG['fed']
            
            rate_score = 4
            for score, criteria in config['ranges'].items():
                if current_rate <= criteria['max_rate']:
                    rate_score = score
                    break
            
            sentiment_adj = config['sentiment_adjustment'].get(sentiment, 0)
            final_score = max(0, min(20, rate_score + sentiment_adj))
            
            explanations = {
                (16, 20): "Accommodative monetary policy supports growth",
                (12, 15): "Neutral policy stance - balanced approach", 
                (8, 11): "Tightening policy may constrain growth",
                (0, 7): "Restrictive policy creating headwinds"
            }
            
            explanation = next((exp for (low, high), exp in explanations.items() 
                              if low <= final_score <= high), "Policy assessment unclear")
            
            return final_score, explanation, metadata
            
        except Exception as e:
            logger.error(f"Error calculating Fed score: {str(e)}")
            return 12, "Error calculating Fed score", {}

    def calculate_earnings_score(self, earnings_df: pd.DataFrame) -> Tuple[int, str, dict]:
        """Calculate earnings score with error handling"""
        try:
            if earnings_df.empty or 'beat_rate' not in earnings_df.columns:
                return 12, "No earnings data available", {}
            
            latest_beat_rate = float(earnings_df['beat_rate'].iloc[-1])
            avg_beat_rate = float(earnings_df['beat_rate'].tail(4).mean())
            trend = ("Improving" if len(earnings_df) > 1 and 
                    latest_beat_rate > earnings_df['beat_rate'].iloc[-2] else "Declining")
            
            metadata = {
                'latest_beat_rate': latest_beat_rate,
                'avg_beat_rate': avg_beat_rate,
                'trend': trend,
                'latest_quarter': earnings_df['quarter'].iloc[-1] if 'quarter' in earnings_df.columns else 'N/A'
            }
            
            config = SCORING_CONFIG['earnings']
            
            if (latest_beat_rate >= config['excellent']['min_beat'] and 
                avg_beat_rate >= config['excellent']['min_avg']):
                return config['excellent']['score'], "Exceptional earnings performance", metadata
            elif latest_beat_rate >= config['good']['min_beat']:
                return config['good']['score'], "Strong earnings beats above average", metadata
            elif latest_beat_rate >= config['fair']['min_beat']:
                return config['fair']['score'], "Solid earnings performance", metadata
            elif latest_beat_rate >= config['poor']['min_beat']:
                return config['poor']['score'], "Mixed earnings results", metadata
            else:
                return config['critical']['score'], "Weak earnings performance", metadata
                
        except Exception as e:
            logger.error(f"Error calculating earnings score: {str(e)}")
            return 12, "Error calculating earnings score", {}

    def calculate_inflation_score(self, inflation_df: pd.DataFrame) -> Tuple[int, str, dict]:
        """Calculate inflation score with error handling"""
        try:
            if inflation_df.empty or 'cpi_yoy' not in inflation_df.columns:
                return 12, "No inflation data available", {}
            
            latest_cpi = float(inflation_df['cpi_yoy'].iloc[-1])
            latest_core = float(inflation_df['core_cpi_yoy'].iloc[-1]) if 'core_cpi_yoy' in inflation_df.columns else latest_cpi
            
            if len(inflation_df) >= 6:
                recent_cpi = inflation_df['cpi_yoy'].tail(3).mean()
                older_cpi = inflation_df['cpi_yoy'].tail(6).head(3).mean()
                trend = "Falling" if recent_cpi < older_cpi else "Rising"
            else:
                trend = "Stable"
            
            metadata = {
                'latest_cpi': latest_cpi,
                'latest_core': latest_core,
                'trend': trend,
                'target_deviation': abs(latest_cpi - 2.0),
                'latest_month': inflation_df.get('month_year', ['N/A']).iloc[-1] if 'month_year' in inflation_df.columns else 'N/A'
            }
            
            config = SCORING_CONFIG['inflation']
            target_deviation = abs(latest_cpi - 2.0)
            core_deviation = abs(latest_core - 2.0)
            
            if (target_deviation <= config['excellent']['max_deviation'] and 
                core_deviation <= config['excellent']['max_core_deviation']):
                return config['excellent']['score'], "Inflation at Fed target", metadata
            elif target_deviation <= config['good']['max_deviation'] and trend == config['good']['trend']:
                return config['good']['score'], "Inflation moving toward target", metadata
            elif target_deviation <= config['fair']['max_deviation']:
                return config['fair']['score'], "Inflation moderately above/below target", metadata
            elif target_deviation <= config['poor']['max_deviation']:
                return config['poor']['score'], "Inflation well above target", metadata
            else:
                return config['critical']['score'], "Inflation crisis - far from target", metadata
                
        except Exception as e:
            logger.error(f"Error calculating inflation score: {str(e)}")
            return 12, "Error calculating inflation score", {}

    def calculate_usd_score(self, usd_df: pd.DataFrame) -> Tuple[int, str, dict]:
        """Calculate USD strength score with error handling"""
        try:
            if usd_df.empty or 'dxy_close' not in usd_df.columns:
                return 12, "No USD data available", {}
            
            latest_dxy = float(usd_df['dxy_close'].iloc[-1])
            avg_dxy_3m = float(usd_df['dxy_close'].tail(3).mean()) if len(usd_df) >= 3 else latest_dxy
            avg_dxy_6m = float(usd_df['dxy_close'].tail(6).mean()) if len(usd_df) >= 6 else avg_dxy_3m
            
            recent_trend = "Rising" if latest_dxy > avg_dxy_3m else "Falling"
            longer_trend = "Rising" if avg_dxy_3m > avg_dxy_6m else "Falling"
            
            metadata = {
                'latest_dxy': latest_dxy,
                'avg_3m': avg_dxy_3m,
                'recent_trend': recent_trend,
                'longer_trend': longer_trend,
                'latest_date': usd_df.get('month_year', ['N/A']).iloc[-1] if 'month_year' in usd_df.columns else 'N/A'
            }
            
            config = SCORING_CONFIG['usd']
            
            if (latest_dxy >= config['excellent']['min_dxy'] and 
                latest_dxy <= config['excellent']['max_dxy'] and 
                recent_trend == "Rising"):
                return config['excellent']['score'], "Optimal USD strength supporting market confidence", metadata
            elif (latest_dxy >= config['good']['min_dxy'] and 
                  latest_dxy <= config['good']['max_dxy']):
                return config['good']['score'], "USD strength within healthy range", metadata
            elif (latest_dxy >= config['fair']['min_dxy'] and 
                  latest_dxy <= config['fair']['max_dxy']):
                return config['fair']['score'], "USD at acceptable levels", metadata
            elif (latest_dxy >= config['poor']['min_dxy'] and 
                  latest_dxy <= config['poor']['max_dxy']):
                if latest_dxy < 95:
                    return config['poor']['score'], "USD weakness may indicate economic concerns", metadata
                else:
                    return config['poor']['score'], "USD strength may pressure exports and growth", metadata
            else:
                if latest_dxy < 90:
                    return config['critical']['score'], "Severe USD weakness indicating potential crisis", metadata
                else:
                    return config['critical']['score'], "Excessive USD strength creating economic imbalances", metadata
                    
        except Exception as e:
            logger.error(f"Error calculating USD score: {str(e)}")
            return 12, "Error calculating USD score", {}

    def calculate_overall_grade(self, scores: Dict[str, int]) -> Tuple[str, int, str, str]:
        """Calculate overall grade using modular scoring system"""
        try:
            total = 0
            total_weight = 0
            
            for metric, score in scores.items():
                if metric in SCORING_CONFIG and score is not None:
                    weight = SCORING_CONFIG[metric].get('weight', 1.0)
                    total += score * weight
                    total_weight += weight * 20
            
            percentage_score = round((total / total_weight) * 100) if total_weight > 0 else 0
            
            config = SCORING_CONFIG['overall_grades']
            for threshold in sorted(config.keys(), reverse=True):
                if percentage_score >= threshold:
                    grade_info = config[threshold]
                    return grade_info['grade'], percentage_score, grade_info['color'], grade_info['sentiment']
            
            return "F", percentage_score, "#ef4444", "negative"
            
        except Exception as e:
            logger.error(f"Error calculating overall grade: {str(e)}")
            return "F", 0, "#ef4444", "negative"

# ============================================================================
# UI COMPONENTS
# ============================================================================

def create_banner_card() -> str:
    """Create professional banner"""
    now = datetime.now()
    return f"""
    <div class="banner-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-weight: 700;">Market Health Dashboard</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Professional Investment Grade Analysis</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; opacity: 0.8;">
                    {now.strftime("%B %d, %Y")}
                </div>
                <div style="font-size: 0.8rem; opacity: 0.7;">
                    Last Updated: {now.strftime("%H:%M:%S")}
                </div>
            </div>
        </div>
    </div>
    """

def create_kpi_card(title: str, value: str, subtitle: str, icon: str, color: str, description: str = "") -> str:
    """Create KPI cards with descriptions"""
    desc_html = f'<div class="kpi-description" style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8; font-style: italic;">{description}</div>' if description else ""
    
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{icon} {title}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
        {desc_html}
    </div>
    """

def create_fallback_warning(fallback_metrics: list) -> str:
    """Create warning card for fallback data usage"""
    if not fallback_metrics:
        return ""
    
    metrics_text = ", ".join(fallback_metrics)
    return f"""
    <div class="data-warning">
        <strong>⚠️ Using fallback data for:</strong> {metrics_text}
        <br><small>Live data unavailable - showing historical/simulated data for analysis</small>
    </div>
    """

def create_error_message(message: str) -> str:
    """Create error message card"""
    return f"""
    <div class="error-message">
        <strong>⚠️ Error:</strong> {message}
    </div>
    """

# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================

def create_gauge_chart(score: int, grade: str, color: str) -> go.Figure:
    """Create professional gauge chart"""
    try:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"<b>Market Health Grade: {grade}</b><br><span style='font-size:0.8em'>Overall Score</span>"},
            delta = {'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#fee2e2'},
                    {'range': [40, 55], 'color': '#fef3c7'},
                    {'range': [55, 70], 'color': '#e0f2fe'},
                    {'range': [70, 85], 'color': '#dbeafe'},
                    {'range': [85, 100], 'color': '#dcfce7'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'color': "darkblue", 'family': "Inter"},
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating gauge chart: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Error creating gauge chart",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=300)
        return fig

def create_enhanced_chart(df: pd.DataFrame, chart_type: str, title: str) -> go.Figure:
    """Create enhanced charts for different metrics"""
    try:
        fig = go.Figure()
        
        if df.empty:
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No data available",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=title,
                height=400,
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return fig
        
        color = COLOR_PALETTE.get(chart_type, '#1f77b4')
        
        if chart_type == 'gdp' and 'gdp_growth' in df.columns and 'date' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['gdp_growth'],
                mode='lines+markers',
                name='GDP Growth Rate',
                line=dict(color=color, width=4),
                marker=dict(size=8, color=color),
                hovertemplate="<b>%{customdata}</b><br>Growth Rate: %{y:.1f}%<extra></extra>",
                customdata=df.get('quarter', df['date'].dt.strftime('%Y-%m-%d'))
            ))
            
            fig.add_hline(y=2.0, line_dash="dash", line_color="red", line_width=2,
                          annotation_text="Long-term Average (2.0%)", 
                          annotation_position="bottom right")
            
            if len(df) > 0:
                latest_value = df['gdp_growth'].iloc[-1]
                fig.add_annotation(
                    x=df['date'].iloc[-1], y=latest_value,
                    text=f"{latest_value:.1f}%",
                    showarrow=True, arrowhead=2, arrowcolor=color,
                    bgcolor="white", bordercolor=color, borderwidth=1
                )
            
            fig.update_layout(yaxis_title="Growth Rate (%)")
            
        elif chart_type == 'fed' and 'fed_rate' in df.columns and 'date' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['fed_rate'],
                mode='lines+markers',
                name='Fed Funds Rate',
                line=dict(color=color, width=4),
                marker=dict(size=8, color=color),
                hovertemplate="<b>%{customdata}</b><br>Fed Rate: %{y:.2f}%<extra></extra>",
                customdata=df.get('month_year', df['date'].dt.strftime('%b %Y'))
            ))
            
            if len(df) > 0:
                latest_value = df['fed_rate'].iloc[-1]
                fig.add_annotation(
                    x=df['date'].iloc[-1], y=latest_value,
                    text=f"{latest_value:.2f}%",
                    showarrow=True, arrowhead=2, arrowcolor=color,
                    bgcolor="white", bordercolor=color, borderwidth=1
                )
            
            fig.update_layout(yaxis_title="Rate (%)")
            
        elif chart_type == 'earnings' and 'quarter' in df.columns and 'beat_rate' in df.columns:
            fig.add_trace(go.Bar(
                x=df['quarter'],
                y=df['beat_rate'],
                name='Beat Rate',
                marker=dict(color=color, opacity=0.85),
                hovertemplate="<b>%{x}</b><br>Beat Rate: %{y}%<extra></extra>"
            ))
            
            if 'miss_rate' in df.columns:
                fig.add_trace(go.Bar(
                    x=df['quarter'],
                    y=df['miss_rate'],
                    name='Miss Rate',
                    marker=dict(color='#d62728', opacity=0.85),
                    hovertemplate="<b>%{x}</b><br>Miss Rate: %{y}%<extra></extra>"
                ))
            
            fig.update_layout(barmode='stack', yaxis_title="Percentage (%)")
            
        elif chart_type == 'inflation' and 'cpi_yoy' in df.columns and 'date' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['cpi_yoy'],
                mode='lines+markers',
                name='CPI YoY',
                line=dict(color=color, width=3),
                marker=dict(size=8, color=color),
                hovertemplate="<b>%{customdata}</b><br>CPI: %{y:.1f}%<extra></extra>",
                customdata=df.get('month_year', df['date'].dt.strftime('%b %Y'))
            ))
            
            if 'core_cpi_yoy' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['core_cpi_yoy'],
                    mode='lines+markers',
                    name='Core CPI YoY',
                    line=dict(color=COLOR_PALETTE['fed'], width=3, dash='dot'),
                    marker=dict(size=8, color=COLOR_PALETTE['fed']),
                    hovertemplate="<b>%{customdata}</b><br>Core CPI: %{y:.1f}%<extra></extra>",
                    customdata=df.get('month_year', df['date'].dt.strftime('%b %Y'))
                ))
            
            fig.add_hline(y=2.0, line_dash="dash", line_color="green",
                          annotation_text="Fed Target 2%", annotation_position="bottom right")
            
            if len(df) > 0:
                latest_val = df['cpi_yoy'].iloc[-1]
                fig.add_annotation(
                    x=df['date'].iloc[-1], y=latest_val,
                    text=f"{latest_val:.1f}%",
                    showarrow=True, arrowhead=2, arrowcolor=color,
                    bgcolor="white", bordercolor=color, borderwidth=1
                )
            
            fig.update_layout(yaxis_title="YoY % Change")
            
        elif chart_type == 'usd' and 'dxy_close' in df.columns and 'date' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['dxy_close'],
                mode='lines+markers',
                name='US Dollar Index',
                line=dict(color=color, width=4),
                marker=dict(size=8, color=color),
                hovertemplate="<b>%{customdata}</b><br>DXY: %{y:.1f}<extra></extra>",
                customdata=df.get('month_year', df['date'].dt.strftime('%b %Y'))
            ))
            
            if len(df) > 0:
                latest_value = df['dxy_close'].iloc[-1]
                fig.add_annotation(
                    x=df['date'].iloc[-1], y=latest_value,
                    text=f"{latest_value:.1f}",
                    showarrow=True, arrowhead=2, arrowcolor=color,
                    bgcolor="white", bordercolor=color, borderwidth=1
                )
            
            fig.update_layout(yaxis_title="Index Value")
        
        else:
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"Required data columns not available for {chart_type}",
                showarrow=False,
                font=dict(size=14)
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            height=400,
            showlegend=(chart_type in ['earnings', 'inflation'])
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating chart for {chart_type}: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error creating {chart_type} chart",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application with comprehensive error handling"""
    try:
        # Load CSS
        st.markdown(load_css(), unsafe_allow_html=True)
        
        # Banner
        st.markdown(create_banner_card(), unsafe_allow_html=True)
        
        # Auto-refresh option (only if streamlit_autorefresh is available)
        if HAS_AUTOREFRESH:
            refresh_interval = st.sidebar.selectbox(
                "Auto-refresh interval (seconds)",
                [0, 30, 60, 300, 900],
                index=0
            )
            
            if refresh_interval > 0:
                st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")
        else:
            st.sidebar.info("Install streamlit_autorefresh for auto-refresh functionality")
        
        # Initialize analyzer
        analyzer = MarketHealthAnalyzer()
        
        # Fetch all data with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Fetching GDP data...")
            progress_bar.progress(20)
            gdp_data = analyzer.fetch_gdp_data()
            
            status_text.text("Fetching Fed data...")
            progress_bar.progress(40)
            fed_data, current_rate, fed_sentiment = analyzer.fetch_fed_data()
            
            status_text.text("Fetching earnings data...")
            progress_bar.progress(60)
            earnings_data = analyzer.fetch_earnings_data()
            
            status_text.text("Fetching inflation data...")
            progress_bar.progress(80)
            inflation_data = analyzer.fetch_inflation_data()
            
            status_text.text("Fetching USD data...")
            progress_bar.progress(90)
            usd_data = analyzer.fetch_usd_data()
            
            status_text.text("Processing data...")
            progress_bar.progress(100)
            
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")
            st.error("Using fallback data for all metrics")
            analyzer.fallback_status = {k: True for k in analyzer.fallback_status}
            
            gdp_data = generate_gdp_fallback_data()
            fed_data, current_rate, fed_sentiment = generate_fed_fallback_data()
            earnings_data = generate_earnings_fallback_data()
            inflation_data = generate_inflation_fallback_data()
            usd_data = generate_usd_fallback_data()
        
        finally:
            progress_bar.empty()
            status_text.empty()
        
        # Calculate scores with error handling
        try:
            gdp_score, gdp_desc, gdp_meta = analyzer.calculate_gdp_score(gdp_data)
            fed_score, fed_desc, fed_meta = analyzer.calculate_fed_score(current_rate, fed_sentiment)
            earnings_score, earnings_desc, earnings_meta = analyzer.calculate_earnings_score(earnings_data)
            inflation_score, inflation_desc, inflation_meta = analyzer.calculate_inflation_score(inflation_data)
            usd_score, usd_desc, usd_meta = analyzer.calculate_usd_score(usd_data)
            
            scores = {
                'gdp': gdp_score,
                'fed': fed_score,
                'earnings': earnings_score,
                'inflation': inflation_score,
                'usd': usd_score
            }
            
            overall_grade, total_score, grade_color, sentiment = analyzer.calculate_overall_grade(scores)
        
        except Exception as e:
            st.error(f"Error calculating scores: {str(e)}")
            st.stop()
        
        # Show fallback warnings if any metrics are using fallback data
        fallback_metrics = [SCORING_CONFIG[k]['name'] for k, v in analyzer.fallback_status.items() if v]
        if fallback_metrics:
            st.markdown(create_fallback_warning(fallback_metrics), unsafe_allow_html=True)
        
        # Display overall grade
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="grade-gauge-container">', unsafe_allow_html=True)
            fig_gauge = create_gauge_chart(total_score, overall_grade, grade_color)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # KPI Cards with descriptions
        st.markdown("### Key Economic Indicators")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        
        with kpi_col1:
            gdp_latest = gdp_meta.get('latest_growth', 0)
            st.markdown(create_kpi_card(
                "GDP Growth", 
                f"{gdp_latest:.1f}%", 
                gdp_meta.get('trend', 'N/A'),
                "📈", 
                COLOR_PALETTE['gdp'],
                gdp_desc
            ), unsafe_allow_html=True)
        
        with kpi_col2:
            st.markdown(create_kpi_card(
                "Fed Rate", 
                f"{current_rate:.2f}%", 
                fed_sentiment,
                "🏦", 
                COLOR_PALETTE['fed'],
                fed_desc
            ), unsafe_allow_html=True)
        
        with kpi_col3:
            earnings_latest = earnings_meta.get('latest_beat_rate', 0)
            st.markdown(create_kpi_card(
                "Earnings Beat", 
                f"{earnings_latest:.0f}%", 
                earnings_meta.get('trend', 'N/A'),
                "💼", 
                COLOR_PALETTE['earnings'],
                earnings_desc
            ), unsafe_allow_html=True)
        
        with kpi_col4:
            inflation_latest = inflation_meta.get('latest_cpi', 0)
            st.markdown(create_kpi_card(
                "CPI Inflation", 
                f"{inflation_latest:.1f}%", 
                inflation_meta.get('trend', 'N/A'),
                "💰", 
                COLOR_PALETTE['inflation'],
                inflation_desc
            ), unsafe_allow_html=True)
        
        with kpi_col5:
            usd_latest = usd_meta.get('latest_dxy', 0)
            st.markdown(create_kpi_card(
                "USD Index", 
                f"{usd_latest:.1f}", 
                usd_meta.get('recent_trend', 'N/A'),
                "💵", 
                COLOR_PALETTE['usd'],
                usd_desc
            ), unsafe_allow_html=True)
        
        # Charts
        st.markdown("### Detailed Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.plotly_chart(
                create_enhanced_chart(gdp_data, 'gdp', "U.S. GDP Growth Rate (Quarterly, Annualized)"),
                use_container_width=True
            )
            
            st.plotly_chart(
                create_enhanced_chart(earnings_data, 'earnings', "S&P 500 Earnings Beat vs Miss Rate"),
                use_container_width=True
            )
            
            st.plotly_chart(
                create_enhanced_chart(usd_data, 'usd', "US Dollar Index (DXY) Trend"),
                use_container_width=True
            )
        
        with chart_col2:
            st.plotly_chart(
                create_enhanced_chart(fed_data, 'fed', "Federal Funds Rate Trend"),
                use_container_width=True
            )
            
            st.plotly_chart(
                create_enhanced_chart(inflation_data, 'inflation', "U.S. Inflation Trends (CPI vs Core CPI)"),
                use_container_width=True
            )
        
        # Score breakdown
        st.markdown("### Score Breakdown")
        
        try:
            score_data = pd.DataFrame({
                'Metric': ['GDP Growth', 'Fed Policy', 'Earnings', 'Inflation', 'USD Strength'],
                'Score': [gdp_score, fed_score, earnings_score, inflation_score, usd_score],
                'Max': [20, 20, 20, 20, 20],
                'Description': [gdp_desc, fed_desc, earnings_desc, inflation_desc, usd_desc]
            })
            
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Bar(
                x=score_data['Metric'],
                y=score_data['Score'],
                marker_color=[COLOR_PALETTE[k] for k in ['gdp', 'fed', 'earnings', 'inflation', 'usd']],
                text=score_data['Score'],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Score: %{y}/20<br>%{customdata}<extra></extra>",
                customdata=score_data['Description']
            ))
            
            fig_scores.update_layout(
                title="Individual Metric Scores",
                xaxis_title="Metrics",
                yaxis_title="Score (out of 20)",
                height=400,
                yaxis=dict(range=[0, 20])
            )
            
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # Summary table
            st.markdown("### Summary Table")
            st.dataframe(
                score_data[['Metric', 'Score', 'Description']],
                use_container_width=True,
                hide_index=True
            )
            
        except Exception as e:
            st.error(f"Error creating score breakdown: {str(e)}")
            st.markdown(create_error_message(f"Could not create score breakdown: {str(e)}"), unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown(
            f"**Market Health Grade: {overall_grade}** | "
            f"**Total Score: {total_score}/100** | "
            f"**Market Sentiment: {sentiment.title()}**"
        )
        
        # API Key setup instructions and dependencies
        with st.sidebar:
            st.markdown("### Setup Instructions")
            st.info("""
            **Required Dependencies:**
            ```bash
            pip install streamlit pandas plotly requests
            ```
            
            **Optional Dependencies:**
            ```bash
            pip install yfinance streamlit_autorefresh
            ```
            
            **API Keys (hardcoded):**
            - FRED API Key: Embedded for convenience
            - Alpha Vantage API Key: Embedded for convenience
            
            The dashboard will work with hardcoded keys and fallback data.
            """)
            
            st.markdown("### Data Sources")
            st.markdown("""
            - **GDP**: FRED (Federal Reserve Economic Data)
            - **Fed Rate**: FRED
            - **Inflation**: FRED (CPI & Core CPI)
            - **USD Index**: Multiple sources (FRED, Yahoo Finance, Alpha Vantage)
            - **Earnings**: Fallback data (realistic simulation)
            """)
            
            st.markdown("### Missing Dependencies")
            if not HAS_YFINANCE:
                st.warning("⚠️ yfinance not installed - USD data using fallback")
            if not HAS_AUTOREFRESH:
                st.warning("⚠️ streamlit_autorefresh not installed - auto-refresh disabled")
            
            # Show current fallback status
            if any(analyzer.fallback_status.values()):
                st.markdown("### Current Data Status")
                for metric, is_fallback in analyzer.fallback_status.items():
                    status = "❌ Fallback" if is_fallback else "✅ Live"
                    st.text(f"{SCORING_CONFIG[metric]['name']}: {status}")

    except Exception as e:
        logger.error(f"Critical error in main application: {str(e)}")
        st.error(f"Critical application error: {str(e)}")
        st.error("Please check your setup and try refreshing the page.")
        
        # Show basic fallback interface
        st.markdown("### Emergency Fallback Mode")
        st.warning("The application encountered a critical error. Showing basic interface.")
        
        try:
            analyzer = MarketHealthAnalyzer()
            gdp_data = generate_gdp_fallback_data()
            fed_data, current_rate, fed_sentiment = generate_fed_fallback_data()
            
            if not gdp_data.empty:
                st.line_chart(gdp_data.set_index('date')['gdp_growth'])
            if not fed_data.empty:
                st.line_chart(fed_data.set_index('date')['fed_rate'])
                
        except Exception as fallback_error:
            st.error(f"Even fallback mode failed: {str(fallback_error)}")

if __name__ == "__main__":
    main()
