import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client, Client
from datetime import datetime
import pytz

# --- Configuration ---
# Initialize Flask App
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Initialize Supabase
try:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Successfully connected to Supabase.")
except Exception as e:
    print(f"Error connecting to Supabase: {e}")

# --- Helper Functions ---

def get_stock_data(symbol):
    """Fetches historical stock data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    # Fetch data for the last 5 years
    history = ticker.history(period="5y")
    return history

def process_data(stock_data):
    """Processes raw stock data into weekly statistics and applies the fix."""
    # Resample to weekly data, starting the week on Monday
    weekly_data = stock_data['Close'].resample('W-MON', label='left', closed='left').ohlc()
    weekly_data.dropna(inplace=True)
    weekly_data.index = weekly_data.index.date

    # --- THIS IS THE CRITICAL FIX ---
    # Remove any potential duplicate weeks that might arise from resampling edge cases.
    # This keeps the first occurrence of each week and removes subsequent duplicates.
    weekly_data = weekly_data[~weekly_data.index.duplicated(keep='first')]
    # --------------------------------

    weekly_data['weekly_return_pct'] = (weekly_data['close'] - weekly_data['open']) / weekly_data['open'] * 100
    
    # Calculate rolling metrics
    weekly_data['avg_win_pct'] = weekly_data['weekly_return_pct'].apply(lambda x: x if x > 0 else 0).rolling(window=52, min_periods=1).mean()
    weekly_data['avg_loss_pct'] = weekly_data['weekly_return_pct'].apply(lambda x: abs(x) if x < 0 else 0).rolling(window=52, min_periods=1).mean()
    
    # Avoid division by zero for profit_factor
    weekly_data['profit_factor'] = weekly_data.apply(
        lambda row: row['avg_win_pct'] / row['avg_loss_pct'] if row['avg_loss_pct'] > 0 else float('inf'),
        axis=1
    )

    # Calculate Sharpe Ratio
    rolling_std = weekly_data['weekly_return_pct'].rolling(window=52, min_periods=1).std()
    weekly_data['sharpe_ratio'] = weekly_data['weekly_return_pct'].rolling(window=52, min_periods=1).mean() / rolling_std.replace(0, np.nan) * np.sqrt(52)

    weekly_data.reset_index(inplace=True)
    weekly_data.rename(columns={'index': 'week_start_date'}, inplace=True)
    
    # Convert 'inf' to a large number or None, as JSON doesn't support 'inf'
    weekly_data.replace([np.inf, -np.inf], None, inplace=True)

    return weekly_data.to_dict('records')


def store_results(symbol, weekly_stats):
    """Stores the calculated weekly statistics in the Supabase database."""
    table_name = 'weekly_stock_performance'
    
    # Prepare data for upsert
    records_to_upsert = []
    for week in weekly_stats:
        # Convert date object to ISO 8601 string format for Supabase
        week_start_date_str = week['week_start_date'].isoformat()
        
        records_to_upsert.append({
            'symbol': symbol,
            'week_start_date': week_start_date_str,
            'open': week.get('open'),
            'high': week.get('high'),
            'low': week.get('low'),
            'close': week.get('close'),
            'weekly_return_pct': week.get('weekly_return_pct'),
            'avg_win_pct': week.get('avg_win_pct'),
            'avg_loss_pct': week.get('avg_loss_pct'),
            'profit_factor': week.get('profit_factor'),
            'sharpe_ratio': week.get('sharpe_ratio'),
            'last_updated': datetime.now(pytz.utc).isoformat()
        })
    
    # Use upsert to insert new records or update existing ones
    # on_conflict specifies the unique constraint
    response = supabase.table(table_name).upsert(records_to_upsert, on_conflict='symbol,week_start_date').execute()
    print(f"Upsert response for {symbol}: {response.data}")


# --- API Routes ---

@app.route('/')
def home():
    return "Stock Analysis API is running!"

@app.route('/analyze', methods=['POST'])
def analyze_route():
    data = request.json
    symbol = data.get('symbol').upper()
    if not symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    try:
        print(f"--- Starting analysis for {symbol} ---")
        
        # Step 1: Fetch data from Yahoo Finance
        stock_data = get_stock_data(symbol)
        if stock_data.empty:
            print(f"Error: No data returned from yfinance for {symbol}")
            return jsonify({"error": f"Could not retrieve historical data for symbol: {symbol}"}), 404
        print(f"Step 1/3: Successfully fetched data for {symbol}.")

        # Step 2: Process the data
        weekly_stats = process_data(stock_data)
        print(f"Step 2/3: Successfully processed data for {symbol}.")

        # Step 3: Store results in Supabase
        store_results(symbol, weekly_stats)
        print(f"Step 3/3: Successfully stored results for {symbol} in Supabase.")

        print(f"--- Analysis complete for {symbol} ---")
        return jsonify(weekly_stats)

    except Exception as e:
        print(f"An error occurred while processing symbol {symbol}:")
        import traceback
        traceback.print_exc() # This prints the full error to your logs
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)