import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Streamlit page configuration
st.set_page_config(
    page_title="Market Health Dashboard | Professional Grade",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Scoring thresholds and configuration
SCORING_CONFIG = {
    'gdp': {
        'excellent': {'min_latest': 3.0, 'min_avg': 2.5, 'score': 25},
        'good': {'min_latest': 2.0, 'min_avg': 2.0, 'score': 20},
        'fair': {'min_latest': 1.0, 'min_avg': 0, 'score': 15},
        'poor': {'min_latest': 0, 'min_avg': 0, 'score': 10},
        'critical': {'score': 5}
    },
    'fed': {
        'ranges': {
            25: {'max_rate': 2.0},
            20: {'max_rate': 3.5},
            15: {'max_rate': 5.0},
            10: {'max_rate': 6.0},
            5: {'max_rate': float('inf')}
        },
        'sentiment_adjustment': {'Dovish': 3, 'Hawkish': -3, 'Neutral': 0}
    },
    'earnings': {
        'excellent': {'min_beat': 75, 'min_avg': 72, 'score': 25},
        'good': {'min_beat': 70, 'score': 20},
        'fair': {'min_beat': 65, 'score': 15},
        'poor': {'min_beat': 60, 'score': 10},
        'critical': {'score': 5}
    },
    'inflation': {
        'excellent': {'max_deviation': 0.5, 'max_core_deviation': 0.5, 'score': 25},
        'good': {'max_deviation': 1.0, 'trend': 'Falling', 'score': 20},
        'fair': {'max_deviation': 1.5, 'score': 15},
        'poor': {'max_deviation': 2.5, 'score': 10},
        'critical': {'score': 5}
    },
    'overall_grades': {
        85: {'grade': 'A', 'color': '#10b981', 'sentiment': 'positive'},
        70: {'grade': 'B', 'color': '#3b82f6', 'sentiment': 'positive'},
        55: {'grade': 'C', 'color': '#f59e0b', 'sentiment': 'neutral'},
        40: {'grade': 'D', 'color': '#f97316', 'sentiment': 'negative'},
        0: {'grade': 'F', 'color': '#ef4444', 'sentiment': 'negative'}
    }
}

# Color palette for charts
COLOR_PALETTE = {
    'gdp': '#1f77b4',
    'fed': '#ff7f0e', 
    'earnings': '#2ca02c',
    'inflation': '#d62728'
}

# ============================================================================
# STYLING AND CSS
# ============================================================================

def load_css():
    """Minimal CSS for essential styling - relies on Streamlit's native theming"""
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
        position: relative;
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
    </style>
    """

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

class MarketHealthAnalyzer:
    """Market health analysis and scoring system with caching"""
    
    def __init__(self):
        self.fred_api_key = "6d9f556dbe4a84d6bf65a4833caa4aa9"  # Replace with your FRED API key
        
    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def fetch_gdp_data(_self):
        """Fetch Real GDP % change (annualized, quarterly) from FRED"""
        try:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "A191RL1Q225SBEA",
                "api_key": _self.fred_api_key,
                "file_type": "json",
                "observation_start": start_date,
                "sort_order": "desc"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "observations" not in data:
                raise ValueError("No observations in FRED response")

            gdp_data = pd.DataFrame(data["observations"])
            gdp_data["date"] = pd.to_datetime(gdp_data["date"])
            gdp_data["gdp_growth"] = pd.to_numeric(gdp_data["value"], errors="coerce")
            gdp_data = gdp_data.dropna().sort_values('date')
            gdp_data["quarter"] = gdp_data["date"].dt.to_period("Q").astype(str)

            return gdp_data.tail(12)
        except Exception as e:
            st.warning(f"Using fallback GDP data due to API error: {str(e)}")
            # Realistic fallback data
            dates = pd.date_range(start='2022-01-01', end='2024-07-01', freq='Q')
            growth_rates = [2.1, 1.8, 2.9, 2.6, 4.2, 2.1, 3.2, 4.9, 2.4, 2.8]
            return pd.DataFrame({
                'date': dates[:len(growth_rates)],
                'gdp_growth': growth_rates,
                'quarter': [f"Q{d.quarter} {d.year}" for d in dates[:len(growth_rates)]]
            })

    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def fetch_fed_data(_self):
        """Fetch Federal Reserve Effective Funds Rate from FRED"""
        try:
            start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
            
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "FEDFUNDS",
                "api_key": _self.fred_api_key,
                "file_type": "json",
                "observation_start": start_date,
                "sort_order": "desc"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "observations" not in data:
                raise ValueError("No observations in FRED response")

            fed_data = pd.DataFrame(data["observations"])
            fed_data["date"] = pd.to_datetime(fed_data["date"])
            fed_data["fed_rate"] = pd.to_numeric(fed_data["value"], errors="coerce")
            fed_data = fed_data.dropna().sort_values('date').tail(60)
            fed_data["month_year"] = fed_data["date"].dt.strftime("%b %Y")

            current_rate = fed_data["fed_rate"].iloc[-1]
            
            # Sentiment analysis based on recent trend
            recent_rates = fed_data["fed_rate"].tail(6)
            rate_change = recent_rates.iloc[-1] - recent_rates.iloc[0]
            
            if rate_change < -0.25:
                sentiment = "Dovish"
            elif rate_change > 0.25:
                sentiment = "Hawkish"
            else:
                sentiment = "Neutral"

            return fed_data, current_rate, sentiment
        except Exception as e:
            st.warning(f"Using fallback Fed data due to API error: {str(e)}")
            # Realistic fallback data with recent Fed policy
            dates = pd.date_range(start='2020-01-01', end='2024-08-01', freq='M')
            rates = [1.75, 1.0, 0.25] + [0.25]*15 + [0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.25, 5.5] + [5.5]*4 + [5.25, 5.0, 4.75, 4.5, 4.25, 4.0, 3.75] + [3.5]*8
            fallback_data = pd.DataFrame({
                'date': dates[:len(rates)],
                'fed_rate': rates,
                'month_year': [d.strftime('%b %Y') for d in dates[:len(rates)]]
            })
            return fallback_data, 3.5, "Neutral"

    @st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
    def fetch_earnings_data(_self):
        """Fetch comprehensive S&P 500 earnings data using multiple APIs"""
        
        # API configurations - add your keys here
        apis = {
            'alpha_vantage': {
                'key': '5SY22TM7S2Q4XEFW',
                'calls_per_minute': 5,
                'delay': 12  # seconds between calls
            },
            'financial_modeling_prep': {
                'key': '6HnbeJevDKDFC0U9mNzQVvoNt9Y12bSj',  # Your FMP API key
                'calls_per_minute': 250,
                'delay': 0.25
            },
            'polygon': {
                'key': 'YOUR_POLYGON_API_KEY',  # Get free key at polygon.io
                'calls_per_minute': 5,
                'delay': 12
            },
            'iex_cloud': {
                'key': 'YOUR_IEX_API_KEY',  # Get free key at iexcloud.io
                'calls_per_minute': 100,
                'delay': 0.6
            }
        }
        
        try:
            # First, get the complete S&P 500 constituent list
            sp500_companies = _self._get_sp500_constituents()
            
            if not sp500_companies:
                raise ValueError("Could not retrieve S&P 500 constituents")
            
            st.info(f"Processing earnings for {len(sp500_companies)} S&P 500 companies...")
            
            all_earnings_data = []
            companies_processed = 0
            
            # Process companies using multiple APIs
            for i, symbol in enumerate(sp500_companies):
                try:
                    earnings_data = None
                    
                    # Try Alpha Vantage first (your working API)
                    if i < 50:  # Use Alpha Vantage for first 50 companies
                        earnings_data = _self._fetch_from_alpha_vantage(symbol, apis['alpha_vantage'])
                        if earnings_data:
                            all_earnings_data.extend(earnings_data)
                            companies_processed += 1
                    
                    # Try Financial Modeling Prep for remaining companies
                    elif i < 300 and apis['financial_modeling_prep']['key'] != 'YOUR_FMP_API_KEY':
                        earnings_data = _self._fetch_from_fmp(symbol, apis['financial_modeling_prep'])
                        if earnings_data:
                            all_earnings_data.extend(earnings_data)
                            companies_processed += 1
                    
                    # Try other APIs for remaining companies
                    elif apis['iex_cloud']['key'] != 'YOUR_IEX_API_KEY':
                        earnings_data = _self._fetch_from_iex(symbol, apis['iex_cloud'])
                        if earnings_data:
                            all_earnings_data.extend(earnings_data)
                            companies_processed += 1
                    
                    # Rate limiting
                    if earnings_data:
                        import time
                        time.sleep(0.5)  # Conservative delay
                    
                    # Update progress every 25 companies
                    if (i + 1) % 25 == 0:
                        st.info(f"Processed {companies_processed} companies so far...")
                    
                except Exception as e:
                    continue
            
            if all_earnings_data:
                # Process the collected data into quarterly beat rates
                quarterly_results = _self._process_earnings_data(all_earnings_data)
                
                st.success(f"Successfully processed {companies_processed} out of {len(sp500_companies)} S&P 500 companies")
                return quarterly_results
            else:
                raise ValueError("No earnings data collected from any API")
                
        except Exception as e:
            st.warning(f"Multiple API approach failed: {str(e)}. Using Alpha Vantage sample.")
            return _self._fallback_alpha_vantage_sample()

    def _get_sp500_constituents(_self):
        """Get complete S&P 500 constituent list from multiple sources"""
        try:
            # Method 1: Wikipedia S&P 500 list (free, reliable)
            import pandas as pd
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_df = tables[0]  # First table contains the constituents
            symbols = sp500_df['Symbol'].tolist()
            
            # Clean symbols (remove dots, etc.)
            cleaned_symbols = []
            for symbol in symbols:
                # Handle special cases like BRK.B -> BRK-B for some APIs
                cleaned_symbol = str(symbol).replace('.', '-')
                cleaned_symbols.append(cleaned_symbol)
            
            return cleaned_symbols[:500]  # Ensure we don't exceed 500
            
        except Exception:
            # Method 2: Financial Modeling Prep S&P 500 endpoint (if API key available)
            try:
                fmp_key = "YOUR_FMP_API_KEY"  # Replace with actual key
                if fmp_key != "YOUR_FMP_API_KEY":
                    url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={fmp_key}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        return [item['symbol'] for item in data]
            except:
                pass
            
            # Method 3: Fallback to major companies
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ',
                'WMT', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE', 'CRM', 'NFLX',
                'KO', 'PEP', 'TMO', 'COST', 'ABT', 'MRK', 'ACN', 'VZ', 'INTC', 'CSCO',
                'PFE', 'T', 'XOM', 'CVX', 'WFC', 'ABBV', 'AVGO', 'TXN', 'ORCL', 'MDT'
            ]

    def _fetch_from_alpha_vantage(_self, symbol, api_config):
        """Fetch earnings from Alpha Vantage"""
        try:
            url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_config['key']}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'quarterlyEarnings' in data and 'Error Message' not in data:
                    return _self._parse_alpha_vantage_data(symbol, data['quarterlyEarnings'][:8])
            return None
        except:
            return None

    def _fetch_from_fmp(_self, symbol, api_config):
        """Fetch earnings from Financial Modeling Prep"""
        try:
            # Get earnings surprises
            url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{symbol}?apikey={api_config['key']}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    return _self._parse_fmp_data(symbol, data[:8])
            return None
        except:
            return None

    def _fetch_from_iex(_self, symbol, api_config):
        """Fetch earnings from IEX Cloud"""
        try:
            url = f"https://cloud.iexapis.com/stable/stock/{symbol}/earnings/4?token={api_config['key']}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and 'earnings' in data:
                    return _self._parse_iex_data(symbol, data['earnings'])
            return None
        except:
            return None

    def _parse_alpha_vantage_data(_self, symbol, quarterly_data):
        """Parse Alpha Vantage earnings data"""
        results = []
        for quarter_data in quarterly_data:
            try:
                fiscal_end = quarter_data['fiscalDateEnding']
                estimated_eps = quarter_data.get('estimatedEPS', None)
                reported_eps = quarter_data.get('reportedEPS', None)
                
                if (estimated_eps and reported_eps and 
                    estimated_eps != "None" and reported_eps != "None"):
                    
                    est_val = float(estimated_eps)
                    rep_val = float(reported_eps)
                    
                    if est_val != 0:
                        date_obj = pd.to_datetime(fiscal_end)
                        quarter_key = f"Q{date_obj.quarter} {date_obj.year}"
                        beat = rep_val > est_val
                        
                        results.append({
                            'symbol': symbol,
                            'quarter': quarter_key,
                            'beat': beat,
                            'estimated': est_val,
                            'reported': rep_val
                        })
            except:
                continue
        return results

    def _parse_fmp_data(_self, symbol, earnings_data):
        """Parse Financial Modeling Prep earnings data"""
        results = []
        for item in earnings_data:
            try:
                date_obj = pd.to_datetime(item['date'])
                quarter_key = f"Q{date_obj.quarter} {date_obj.year}"
                estimated = float(item['estimatedEarning'])
                actual = float(item['actualEarning'])
                
                if estimated != 0:
                    beat = actual > estimated
                    results.append({
                        'symbol': symbol,
                        'quarter': quarter_key,
                        'beat': beat,
                        'estimated': estimated,
                        'reported': actual
                    })
            except:
                continue
        return results

    def _parse_iex_data(_self, symbol, earnings_data):
        """Parse IEX Cloud earnings data"""
        results = []
        for item in earnings_data:
            try:
                date_obj = pd.to_datetime(item['fiscalEndDate'])
                quarter_key = f"Q{date_obj.quarter} {date_obj.year}"
                estimated = float(item['consensusEPS'] or 0)
                actual = float(item['actualEPS'] or 0)
                
                if estimated != 0:
                    beat = actual > estimated
                    results.append({
                        'symbol': symbol,
                        'quarter': quarter_key,
                        'beat': beat,
                        'estimated': estimated,
                        'reported': actual
                    })
            except:
                continue
        return results

    def _process_earnings_data(_self, all_earnings_data):
        """Process collected earnings data into quarterly beat rates"""
        quarterly_results = {}
        
        for item in all_earnings_data:
            quarter = item['quarter']
            beat = item['beat']
            
            if quarter not in quarterly_results:
                quarterly_results[quarter] = {'total': 0, 'beats': 0}
            
            quarterly_results[quarter]['total'] += 1
            if beat:
                quarterly_results[quarter]['beats'] += 1
        
        # Convert to DataFrame format
        result_data = []
        for quarter, stats in sorted(quarterly_results.items(), 
                                   key=lambda x: (int(x[0].split()[1]), int(x[0][1]))):
            if stats['total'] >= 5:  # Only include quarters with decent sample size
                beat_rate = round((stats['beats'] / stats['total']) * 100)
                miss_rate = 100 - beat_rate
                
                result_data.append({
                    'quarter': quarter,
                    'beat_rate': beat_rate,
                    'miss_rate': miss_rate,
                    'total_companies': stats['total'],
                    'companies_beat': stats['beats'],
                    'companies_missed': stats['total'] - stats['beats']
                })
        
        return pd.DataFrame(result_data).tail(8) if result_data else pd.DataFrame()

    def _fallback_alpha_vantage_sample(_self):
        """Fallback to Alpha Vantage sample if comprehensive approach fails"""
        # Your existing Alpha Vantage logic as fallback
        api_key = "5SY22TM7S2Q4XEFW"
        sample_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
        
        quarterly_results = {}
        companies_processed = 0
        
        for symbol in sample_companies:
            try:
                url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'quarterlyEarnings' in data and 'Error Message' not in data:
                        companies_processed += 1
                        quarterly = data['quarterlyEarnings'][:8]
                        
                        for quarter_data in quarterly:
                            try:
                                fiscal_end = quarter_data['fiscalDateEnding']
                                date_obj = pd.to_datetime(fiscal_end)
                                quarter_key = f"Q{date_obj.quarter} {date_obj.year}"
                                
                                estimated_eps = quarter_data.get('estimatedEPS', None)
                                reported_eps = quarter_data.get('reportedEPS', None)
                                
                                if (estimated_eps and reported_eps and 
                                    estimated_eps != "None" and reported_eps != "None"):
                                    
                                    est_val = float(estimated_eps)
                                    rep_val = float(reported_eps)
                                    
                                    if est_val != 0:
                                        beat = rep_val > est_val
                                        
                                        if quarter_key not in quarterly_results:
                                            quarterly_results[quarter_key] = {'total': 0, 'beats': 0}
                                        
                                        quarterly_results[quarter_key]['total'] += 1
                                        if beat:
                                            quarterly_results[quarter_key]['beats'] += 1
                            except:
                                continue
                
                import time
                time.sleep(0.5)
                
            except:
                continue
        
        # Convert to DataFrame
        result_data = []
        for quarter, stats in sorted(quarterly_results.items()):
            if stats['total'] > 0:
                beat_rate = round((stats['beats'] / stats['total']) * 100)
                result_data.append({
                    'quarter': quarter,
                    'beat_rate': beat_rate,
                    'miss_rate': 100 - beat_rate,
                    'total_companies': stats['total'],
                    'companies_beat': stats['beats'],
                    'companies_missed': stats['total'] - stats['beats']
                })
        
        if result_data:
            st.info(f"Fallback: Processed {companies_processed} major companies")
            return pd.DataFrame(result_data).tail(8)
        
        # Final fallback to historical data
        quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
        beat_rates = [78, 76, 73, 67, 79, 77, 75, 73]
        
        return pd.DataFrame({
            'quarter': quarters,
            'beat_rate': beat_rates,
            'miss_rate': [100-r for r in beat_rates],
            'total_companies': [500] * len(quarters),
            'companies_beat': [int((r/100) * 500) for r in beat_rates],
            'companies_missed': [int(((100-r)/100) * 500) for r in beat_rates]
        })

    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def fetch_inflation_data(_self):
        """Fetch CPI and Core CPI data from FRED"""
        try:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            
            # Fetch CPI data
            cpi_url = "https://api.stlouisfed.org/fred/series/observations"
            cpi_params = {
                "series_id": "CPIAUCSL",
                "api_key": _self.fred_api_key,
                "file_type": "json",
                "units": "pc1",
                "observation_start": start_date,
                "sort_order": "desc"
            }
            cpi_response = requests.get(cpi_url, params=cpi_params, timeout=10)
            cpi_response.raise_for_status()
            cpi_data = cpi_response.json()
            
            # Fetch Core CPI data
            core_params = cpi_params.copy()
            core_params["series_id"] = "CPILFESL"
            core_response = requests.get(cpi_url, params=core_params, timeout=10)
            core_response.raise_for_status()
            core_data = core_response.json()
            
            if "observations" not in cpi_data or "observations" not in core_data:
                raise ValueError("No observations in FRED response")
            
            # Process and merge data
            cpi_df = pd.DataFrame(cpi_data["observations"])
            cpi_df["date"] = pd.to_datetime(cpi_df["date"])
            cpi_df["cpi_yoy"] = pd.to_numeric(cpi_df["value"], errors="coerce")
            
            core_df = pd.DataFrame(core_data["observations"])
            core_df["date"] = pd.to_datetime(core_df["date"])
            core_df["core_cpi_yoy"] = pd.to_numeric(core_df["value"], errors="coerce")
            
            inflation_df = pd.merge(cpi_df[['date', 'cpi_yoy']], 
                                  core_df[['date', 'core_cpi_yoy']], 
                                  on='date', how='inner')
            
            inflation_df = inflation_df.dropna().sort_values('date')
            inflation_df["month_year"] = inflation_df["date"].dt.strftime("%b %Y")
            
            return inflation_df.tail(24)
            
        except Exception as e:
            st.warning(f"Using fallback inflation data due to API error: {str(e)}")
            # Recent inflation trends
            dates = pd.date_range(start='2022-08-01', end='2024-08-01', freq='M')
            cpi_data = [8.3, 8.2, 7.7, 6.5, 6.0, 6.4, 5.0, 4.9, 4.0, 3.2, 3.0, 3.7, 
                       3.2, 3.1, 2.4, 2.6, 2.4, 3.5, 3.4, 3.3, 3.2, 2.9, 2.6, 2.4, 2.5]
            core_cpi_data = [6.3, 6.6, 6.0, 5.5, 5.7, 5.3, 5.2, 4.8, 4.0, 3.8, 4.1, 3.6, 
                            4.0, 3.2, 3.3, 3.2, 3.8, 3.6, 3.4, 3.3, 3.2, 3.2, 3.3, 3.2, 3.3]
            
            return pd.DataFrame({
                'date': dates[:len(cpi_data)],
                'cpi_yoy': cpi_data,
                'core_cpi_yoy': core_cpi_data,
                'month_year': [d.strftime('%b %Y') for d in dates[:len(cpi_data)]]
            })

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

    def calculate_gdp_score(self, gdp_df):
        """Calculate GDP score using configuration"""
        if gdp_df.empty:
            return 15, "No data available", {}
        
        latest_growth = gdp_df['gdp_growth'].iloc[-1]
        avg_growth = gdp_df['gdp_growth'].tail(4).mean()
        trend = "Rising" if len(gdp_df) > 1 and latest_growth > gdp_df['gdp_growth'].iloc[-2] else "Falling"
        
        metadata = {
            'latest_growth': latest_growth,
            'avg_growth': avg_growth,
            'trend': trend,
            'latest_quarter': gdp_df['quarter'].iloc[-1] if 'quarter' in gdp_df.columns else 'N/A'
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

    def calculate_fed_score(self, current_rate, sentiment):
        """Calculate Fed score using configuration"""
        if current_rate is None:
            return 15, "No Fed data available", {}
            
        metadata = {
            'current_rate': current_rate,
            'sentiment': sentiment,
            'policy_stance': 'Restrictive' if current_rate > 4.5 else 'Neutral' if current_rate > 2.5 else 'Accommodative'
        }
        
        config = SCORING_CONFIG['fed']
        
        # Find appropriate score based on rate ranges
        rate_score = 5  # default
        for score, criteria in config['ranges'].items():
            if current_rate <= criteria['max_rate']:
                rate_score = score
                break
        
        # Apply sentiment adjustment
        sentiment_adj = config['sentiment_adjustment'].get(sentiment, 0)
        final_score = max(0, min(25, rate_score + sentiment_adj))
        
        explanations = {
            (20, 25): "Accommodative monetary policy supports growth",
            (15, 19): "Neutral policy stance - balanced approach", 
            (10, 14): "Tightening policy may constrain growth",
            (0, 9): "Restrictive policy creating headwinds"
        }
        
        explanation = next((exp for (low, high), exp in explanations.items() if low <= final_score <= high), 
                          "Policy assessment unclear")
        
        return final_score, explanation, metadata

    def calculate_earnings_score(self, earnings_df):
        """Calculate earnings score using configuration"""
        if earnings_df.empty:
            return 15, "No earnings data available", {}
        
        latest_beat_rate = earnings_df['beat_rate'].iloc[-1]
        avg_beat_rate = earnings_df['beat_rate'].tail(4).mean()
        trend = "Improving" if len(earnings_df) > 1 and latest_beat_rate > earnings_df['beat_rate'].iloc[-2] else "Declining"
        
        metadata = {
            'latest_beat_rate': latest_beat_rate,
            'avg_beat_rate': avg_beat_rate,
            'trend': trend,
            'latest_quarter': earnings_df['quarter'].iloc[-1]
        }
        
        config = SCORING_CONFIG['earnings']
        
        if latest_beat_rate >= config['excellent']['min_beat'] and avg_beat_rate >= config['excellent']['min_avg']:
            return config['excellent']['score'], "Exceptional earnings performance", metadata
        elif latest_beat_rate >= config['good']['min_beat']:
            return config['good']['score'], "Strong earnings beats above average", metadata
        elif latest_beat_rate >= config['fair']['min_beat']:
            return config['fair']['score'], "Solid earnings performance", metadata
        elif latest_beat_rate >= config['poor']['min_beat']:
            return config['poor']['score'], "Mixed earnings results", metadata
        else:
            return config['critical']['score'], "Weak earnings performance", metadata

    def calculate_inflation_score(self, inflation_df):
        """Calculate inflation score using configuration"""
        if inflation_df.empty:
            return 15, "No inflation data available", {}
        
        latest_cpi = inflation_df['cpi_yoy'].iloc[-1]
        latest_core = inflation_df['core_cpi_yoy'].iloc[-1]
        
        recent_cpi = inflation_df['cpi_yoy'].tail(3).mean()
        older_cpi = inflation_df['cpi_yoy'].tail(6).head(3).mean()
        trend = "Falling" if recent_cpi < older_cpi else "Rising"
        
        metadata = {
            'latest_cpi': latest_cpi,
            'latest_core': latest_core,
            'trend': trend,
            'target_deviation': abs(latest_cpi - 2.0),
            'latest_month': inflation_df['month_year'].iloc[-1] if 'month_year' in inflation_df.columns else 'N/A'
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

    def calculate_overall_grade(self, scores):
        """Calculate overall grade using configuration"""
        total = sum(scores.values())
        
        config = SCORING_CONFIG['overall_grades']
        for threshold in sorted(config.keys(), reverse=True):
            if total >= threshold:
                grade_info = config[threshold]
                return grade_info['grade'], total, grade_info['color'], grade_info['sentiment']
        
        # Fallback
        return "F", total, "#ef4444", "negative"

# ============================================================================
# UI COMPONENTS
# ============================================================================

def create_banner_card():
    """Create professional banner with status indicators"""
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

def create_kpi_card(title, value, subtitle, icon, color):
    """Create KPI cards with improved styling"""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{icon} {title}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """

# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================

def create_gauge_chart(score, grade, color):
    """Create professional circular gauge for overall grade"""
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

def create_enhanced_gdp_chart(gdp_df):
    """Create enhanced GDP chart"""
    fig = go.Figure()
    
    if not gdp_df.empty and 'date' in gdp_df.columns and 'gdp_growth' in gdp_df.columns:
        fig.add_trace(go.Scatter(
            x=gdp_df['date'],
            y=gdp_df['gdp_growth'],
            mode='lines+markers',
            name='GDP Growth Rate',
            line=dict(color=COLOR_PALETTE['gdp'], width=4),
            marker=dict(size=8, color=COLOR_PALETTE['gdp']),
            hovertemplate="<b>%{customdata}</b><br>Growth Rate: %{y:.1f}%<extra></extra>",
            customdata=gdp_df['quarter'] if 'quarter' in gdp_df.columns else gdp_df['date'].dt.strftime('%Q%q %Y')
        ))
        
        # Add target line
        fig.add_hline(y=2.0, line_dash="dash", line_color="red", line_width=2,
                      annotation_text="Long-term Average (2.0%)", 
                      annotation_position="bottom right")
        
        # Annotate latest point
        latest_value = gdp_df['gdp_growth'].iloc[-1]
        fig.add_annotation(
            x=gdp_df['date'].iloc[-1],
            y=latest_value,
            text=f"{latest_value:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLOR_PALETTE['gdp'],
            bgcolor="white",
            bordercolor=COLOR_PALETTE['gdp'],
            borderwidth=1
        )
    
    fig.update_layout(
        title="U.S. GDP Growth Rate (Quarterly, Annualized)",
        xaxis_title="Date",
        yaxis_title="Growth Rate (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_enhanced_fed_chart(fed_df):
    """Create enhanced Federal Reserve rate chart"""
    fig = go.Figure()
    
    if not fed_df.empty and 'date' in fed_df.columns and 'fed_rate' in fed_df.columns:
        fig.add_trace(go.Scatter(
            x=fed_df['date'],
            y=fed_df['fed_rate'],
            mode='lines+markers',
            name='Fed Funds Rate',
            line=dict(color=COLOR_PALETTE['fed'], width=4),
            marker=dict(size=8, color=COLOR_PALETTE['fed']),
            hovertemplate="<b>%{customdata}</b><br>Fed Rate: %{y:.2f}%<extra></extra>",
            customdata=fed_df['month_year'] if 'month_year' in fed_df.columns else fed_df['date'].dt.strftime('%b %Y')
        ))
        
        # Annotate latest point
        latest_value = fed_df['fed_rate'].iloc[-1]
        fig.add_annotation(
            x=fed_df['date'].iloc[-1],
            y=latest_value,
            text=f"{latest_value:.2f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLOR_PALETTE['fed'],
            bgcolor="white",
            bordercolor=COLOR_PALETTE['fed'],
            borderwidth=1
        )
    
    fig.update_layout(
        title="Federal Funds Rate Trend",
        xaxis_title="Date",
        yaxis_title="Rate (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_enhanced_earnings_chart(earnings_df):
    """Create enhanced earnings beat/miss chart"""
    fig = go.Figure()
    
    if not earnings_df.empty and all(col in earnings_df.columns for col in ['quarter', 'beat_rate', 'miss_rate']):
        # Beat Rate Bar
        fig.add_trace(go.Bar(
            x=earnings_df['quarter'],
            y=earnings_df['beat_rate'],
            name='Beat Rate',
            marker=dict(color=COLOR_PALETTE['earnings'], opacity=0.85),
            hovertemplate="<b>%{x}</b><br>Beat Rate: %{y}%<extra></extra>"
        ))
        
        # Miss Rate Bar
        fig.add_trace(go.Bar(
            x=earnings_df['quarter'],
            y=earnings_df['miss_rate'],
            name='Miss Rate',
            marker=dict(color='#d62728', opacity=0.85),
            hovertemplate="<b>%{x}</b><br>Miss Rate: %{y}%<extra></extra>"
        ))

        # Annotate latest point
        latest_q = earnings_df['quarter'].iloc[-1]
        latest_beat = earnings_df['beat_rate'].iloc[-1]
        fig.add_annotation(
            x=latest_q,
            y=latest_beat,
            text=f"{latest_beat}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLOR_PALETTE['earnings'],
            bgcolor="white",
            bordercolor=COLOR_PALETTE['earnings'],
            borderwidth=1
        )
    
    fig.update_layout(
        title="S&P 500 Earnings Beat vs Miss Rate",
        barmode='stack',
        height=400,
        xaxis=dict(title="Quarter"),
        yaxis=dict(title="Percentage (%)")
    )
    
    return fig

def create_enhanced_inflation_chart(inflation_df):
    """Create enhanced inflation chart with CPI and Core CPI trends"""
    fig = go.Figure()
    
    if not inflation_df.empty:
        fig.add_trace(go.Scatter(
            x=inflation_df['date'],
            y=inflation_df['cpi_yoy'],
            mode='lines+markers',
            name='CPI YoY',
            line=dict(color=COLOR_PALETTE['inflation'], width=3),
            marker=dict(size=8, color=COLOR_PALETTE['inflation']),
            hovertemplate="<b>%{customdata}</b><br>CPI: %{y:.1f}%<extra></extra>",
            customdata=inflation_df['month_year']
        ))
        fig.add_trace(go.Scatter(
            x=inflation_df['date'],
            y=inflation_df['core_cpi_yoy'],
            mode='lines+markers',
            name='Core CPI YoY',
            line=dict(color=COLOR_PALETTE['fed'], width=3, dash='dot'),
            marker=dict(size=8, color=COLOR_PALETTE['fed']),
            hovertemplate="<b>%{customdata}</b><br>Core CPI: %{y:.1f}%<extra></extra>",
            customdata=inflation_df['month_year']
        ))

        # Fed target line
        fig.add_hline(y=2.0, line_dash="dash", line_color="green",
                      annotation_text="Fed Target 2%", annotation_position="bottom right")

        # Latest annotation
        latest_val = inflation_df['cpi_yoy'].iloc[-1]
        fig.add_annotation(
            x=inflation_df['date'].iloc[-1],
            y=latest_val,
            text=f"{latest_val:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLOR_PALETTE['inflation'],
            bgcolor="white",
            bordercolor=COLOR_PALETTE['inflation'],
            borderwidth=1
        )
    
    fig.update_layout(
        title="U.S. Inflation Trends (CPI vs Core CPI)",
        height=400,
        xaxis=dict(title="Date"),
        yaxis=dict(title="YoY % Change")
    )
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Load CSS
    st.markdown(load_css(), unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR CONFIGURATION
    # ========================================================================
    st.sidebar.title("Dashboard Settings")
    
    # Auto-refresh with st_autorefresh (non-blocking)
    refresh_interval = st.sidebar.selectbox(
        "Auto Refresh Interval",
        options=[0, 30, 60, 300],
        format_func=lambda x: "Disabled" if x == 0 else f"{x} seconds",
        index=0
    )
    
    if refresh_interval > 0:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")
    
    # Methodology information
    st.sidebar.markdown("### Methodology")
    st.sidebar.write("""
    **GDP**: Score based on latest quarterly growth and 4Q average from FRED.  
    **Fed**: Score based on Fed Funds Rate and recent trend analysis.  
    **Earnings**: Estimated % of S&P 500 companies beating EPS estimates.  
    **Inflation**: Live CPI vs Core CPI vs Fed 2% target from FRED.  
    """)
    st.sidebar.markdown("---")
    st.sidebar.write("*Data sources: FRED API, Alpha Vantage API*")
    st.sidebar.write("*FRED data may have 1-3 month delays*")
    
    # ========================================================================
    # HEADER AND BANNER
    # ========================================================================
    st.markdown(create_banner_card(), unsafe_allow_html=True)
    
    # ========================================================================
    # DATA FETCHING
    # ========================================================================
    with st.spinner("Loading market data..."):
        analyzer = MarketHealthAnalyzer()
        
        # Fetch all data using cached methods
        gdp_df = analyzer.fetch_gdp_data()
        fed_df, current_rate, fed_sent = analyzer.fetch_fed_data()
        earnings_df = analyzer.fetch_earnings_data()
        infl_df = analyzer.fetch_inflation_data()

    # ========================================================================
    # SCORING CALCULATIONS
    # ========================================================================
    gdp_score, gdp_exp, gdp_meta = analyzer.calculate_gdp_score(gdp_df)
    fed_score, fed_exp, fed_meta = analyzer.calculate_fed_score(current_rate, fed_sent)
    earn_score, earn_exp, earn_meta = analyzer.calculate_earnings_score(earnings_df)
    infl_score, infl_exp, infl_meta = analyzer.calculate_inflation_score(infl_df)
    
    scores = {'GDP': gdp_score, 'Fed': fed_score, 'Earnings': earn_score, 'Inflation': infl_score}
    grade, total, grade_color, sentiment = analyzer.calculate_overall_grade(scores)

    # ========================================================================
    # DATA FRESHNESS INDICATORS
    # ========================================================================
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if not gdp_df.empty and 'date' in gdp_df.columns:
            latest_gdp_date = gdp_df['date'].iloc[-1].strftime('%Y-%m-%d')
            st.caption(f"GDP: Latest {latest_gdp_date}")
        else:
            st.caption("GDP: No data")
    with col2:
        if not fed_df.empty and 'date' in fed_df.columns:
            latest_fed_date = fed_df['date'].iloc[-1].strftime('%Y-%m-%d')
            st.caption(f"Fed: Latest {latest_fed_date}")
        else:
            st.caption("Fed: No data")
    with col3:
        st.caption("Earnings: Alpha Vantage API - Sample of Major S&P 500 Companies")
    with col4:
        if not infl_df.empty and 'date' in infl_df.columns:
            latest_infl_date = infl_df['date'].iloc[-1].strftime('%Y-%m-%d')
            st.caption(f"CPI: Latest {latest_infl_date}")
        else:
            st.caption("CPI: No data")

    # Data quality warning for earnings
    st.markdown("""
    <div class="data-warning">
        <strong>Note:</strong> Earnings data is sourced from Alpha Vantage API using a sample of major 
        S&P 500 companies (AAPL, MSFT, GOOGL). For complete market coverage, consider upgrading 
        to a comprehensive financial data provider.
        <br><br>
        <strong>Setup Required:</strong> Get your free Alpha Vantage API key at alphavantage.co 
        and replace "YOUR_ALPHA_VANTAGE_API_KEY" in the code.
    </div>
    """, unsafe_allow_html=True)

    # ========================================================================
    # KPI CARDS LAYOUT
    # ========================================================================
    st.markdown("<div style='display:flex; gap:1rem;'>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gdp_latest = gdp_meta.get('latest_growth', 0)
        gdp_subtitle = f"Latest: {gdp_latest:.1f}%" if gdp_latest is not None else "Latest: N/A"
        st.markdown(create_kpi_card("GDP Growth", f"{gdp_score}/25", gdp_subtitle, "📈", COLOR_PALETTE['gdp']), 
                   unsafe_allow_html=True)
    
    with col2:
        rate_subtitle = f"Rate: {current_rate:.2f}%" if current_rate is not None else "Rate: N/A"
        st.markdown(create_kpi_card("Federal Reserve", f"{fed_score}/25", rate_subtitle, "🏦", COLOR_PALETTE['fed']), 
                   unsafe_allow_html=True)
    
    with col3:
        beat_rate = earn_meta.get('latest_beat_rate', 0)
        beat_subtitle = f"Beat Rate: {beat_rate}%" if beat_rate is not None else "Beat Rate: N/A"
        st.markdown(create_kpi_card("Earnings", f"{earn_score}/25", beat_subtitle, "💰", COLOR_PALETTE['earnings']), 
                   unsafe_allow_html=True)
    
    with col4:
        cpi_latest = infl_meta.get('latest_cpi', 0)
        cpi_subtitle = f"CPI: {cpi_latest:.1f}%" if cpi_latest is not None else "CPI: N/A"
        st.markdown(create_kpi_card("Inflation", f"{infl_score}/25", cpi_subtitle, "📊", COLOR_PALETTE['inflation']), 
                   unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # OVERALL GRADE GAUGE
    # ========================================================================
    st.markdown('<div class="grade-gauge-container">', unsafe_allow_html=True)
    st.plotly_chart(create_gauge_chart(total, grade, grade_color), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================================
    # CHARTS LAYOUT (2x2 GRID)
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_enhanced_gdp_chart(gdp_df), use_container_width=True)
        st.plotly_chart(create_enhanced_earnings_chart(earnings_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_enhanced_fed_chart(fed_df), use_container_width=True)
        st.plotly_chart(create_enhanced_inflation_chart(infl_df), use_container_width=True)

    # ========================================================================
    # DETAILED ANALYSIS SECTION
    # ========================================================================
    with st.expander("📊 Detailed Analysis"):
        st.subheader("Key Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Economic Growth**")
            gdp_growth = gdp_meta.get('latest_growth', None)
            st.write(f"- Latest GDP growth: {gdp_growth:.1f}%" if gdp_growth is not None else "- Latest GDP growth: N/A")
            avg_growth = gdp_meta.get('avg_growth', None)
            st.write(f"- 4-quarter average: {avg_growth:.1f}%" if avg_growth is not None else "- 4-quarter average: N/A")
            st.write(f"- Trend: {gdp_meta.get('trend', 'N/A')}")
            
            st.write("**Monetary Policy**")
            st.write(f"- Current Fed rate: {current_rate:.2f}%" if current_rate is not None else "- Current Fed rate: N/A")
            st.write(f"- Policy stance: {fed_meta.get('policy_stance', 'N/A')}")
            st.write(f"- Market sentiment: {fed_sent}")
        
        with col2:
            st.write("**Corporate Earnings** *(Alpha Vantage Sample)*")
            beat_rate = earn_meta.get('latest_beat_rate', None)
            st.write(f"- Latest beat rate: {beat_rate}%" if beat_rate is not None else "- Latest beat rate: N/A")
            avg_beat = earn_meta.get('avg_beat_rate', None)
            st.write(f"- 4-quarter average: {avg_beat:.1f}%" if avg_beat is not None else "- 4-quarter average: N/A")
            st.write(f"- Trend: {earn_meta.get('trend', 'N/A')}")
            
            st.write("**Price Stability**")
            latest_cpi = infl_meta.get('latest_cpi', None)
            st.write(f"- Current CPI: {latest_cpi:.1f}%" if latest_cpi is not None else "- Current CPI: N/A")
            latest_core = infl_meta.get('latest_core', None)
            st.write(f"- Core CPI: {latest_core:.1f}%" if latest_core is not None else "- Core CPI: N/A")
            st.write(f"- Trend: {infl_meta.get('trend', 'N/A')}")

if __name__ == "__main__":
    main()
