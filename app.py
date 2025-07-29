import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client, Client
from datetime import datetime, timedelta
import pytz
import traceback

# --- Configuration ---
app = Flask(__name__)
CORS(app) 

try:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Successfully connected to Supabase.")
except Exception as e:
    print(f"FATAL: Error connecting to Supabase on startup: {e}")

# --- Helper Functions ---

def get_stock_data(symbol):
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="5y")
    return history

def process_and_analyze_data(stock_data, filter_type='all'):
    """Process stock data and return the structure expected by the frontend."""
    
    # Resample to weekly data
    weekly_data = stock_data['Close'].resample('W-MON', label='left', closed='left').ohlc()
    weekly_data.dropna(inplace=True)
    weekly_data['weekly_return_pct'] = (weekly_data['close'] - weekly_data['open']) / weekly_data['open'] * 100
    
    # Add week number and year
    weekly_data['week_number'] = weekly_data.index.isocalendar().week
    weekly_data['year'] = weekly_data.index.year
    
    # Apply filter if needed
    if filter_type == 'after_up':
        # Shift returns to get previous week's return
        weekly_data['prev_return'] = weekly_data['weekly_return_pct'].shift(1)
        weekly_data = weekly_data[weekly_data['prev_return'] > 0]
    elif filter_type == 'after_down':
        weekly_data['prev_return'] = weekly_data['weekly_return_pct'].shift(1)
        weekly_data = weekly_data[weekly_data['prev_return'] < 0]
    
    # Get current week info
    current_week = datetime.now().isocalendar()[1]
    last_week_data = weekly_data.iloc[-1] if len(weekly_data) > 0 else None
    prior_week_return = weekly_data['weekly_return_pct'].iloc[-2] if len(weekly_data) > 1 else 0
    
    # Calculate statistics by week number
    stats_by_week = []
    for week_num in range(1, 54):
        week_data = weekly_data[weekly_data['week_number'] == week_num]
        if len(week_data) > 0:
            returns = week_data['weekly_return_pct']
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
            avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0
            
            stats_by_week.append({
                'week_number': week_num,
                'count': len(week_data),
                'avg_return': returns.mean(),
                'pct_profitable': (len(positive_returns) / len(returns) * 100) if len(returns) > 0 else 0,
                'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(52) if returns.std() > 0 else 0,
                'std_dev': returns.std()
            })
    
    # Get this year's history
    current_year = datetime.now().year
    this_year_data = weekly_data[weekly_data['year'] == current_year].copy()
    
    history = []
    for idx, row in this_year_data.iterrows():
        history.append({
            'week_number': row['week_number'],
            'year': row['year'],
            'open_price': row['open'],
            'high_price': row['high'],
            'low_price': row['low'],
            'close_price': row['close'],
            'weekly_return_pct': row['weekly_return_pct']
        })
    
    # Store in database
    store_weekly_data(stock_data.index[0].strftime('%Y-%m-%d'), weekly_data)
    
    return {
        'current_info': {
            'current_week': current_week,
            'prior_week_return': prior_week_return
        },
        'statistics': stats_by_week,
        'history': history
    }

def store_weekly_data(symbol, weekly_data):
    """Store processed weekly data in Supabase."""
    table_name = 'weekly_stock_performance'
    records_to_upsert = []
    
    for idx, row in weekly_data.iterrows():
        records_to_upsert.append({
            'symbol': symbol.upper(),
            'week_start_date': idx.date().isoformat(),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'weekly_return_pct': float(row['weekly_return_pct']) if not pd.isna(row['weekly_return_pct']) else None,
            'week_number': int(row['week_number']),
            'year': int(row['year']),
            'last_updated': pytz.utc.localize(datetime.utcnow()).isoformat()
        })
    
    # Only store the last 260 weeks (5 years) to avoid too much data
    records_to_upsert = records_to_upsert[-260:]
    
    if records_to_upsert:
        response = supabase.table(table_name).upsert(records_to_upsert, on_conflict='symbol,week_start_date').execute()
        print(f"Stored {len(records_to_upsert)} weeks of data")

# --- API Routes ---

@app.route('/')
def home():
    return "Stock Analysis API is running!"

@app.route('/test')
def test_route():
    return jsonify({"status": "ok", "message": "The Flask server is running correctly."})

@app.route('/api/analyze/<symbol>')
def analyze_route(symbol):
    """Analyze a stock and return statistics in the format expected by the frontend."""
    symbol = symbol.upper()
    filter_type = request.args.get('filter', 'all')
    
    try:
        print(f"--- Starting analysis for {symbol} with filter: {filter_type} ---")
        
        # Fetch stock data
        stock_data = get_stock_data(symbol)
        if stock_data.empty:
            return jsonify({"error": f"Could not retrieve data for symbol: {symbol}"}), 404
        
        print(f"Fetched {len(stock_data)} days of data for {symbol}")
        
        # Process and analyze
        result = process_and_analyze_data(stock_data, filter_type)
        
        print(f"Analysis complete for {symbol}")
        return jsonify(result)
        
    except Exception as e:
        print(f"ERROR analyzing {symbol}:")
        traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=True)