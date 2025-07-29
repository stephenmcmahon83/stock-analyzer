import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client, Client
from datetime import datetime
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

def process_data(stock_data):
    weekly_data = stock_data['Close'].resample('W-MON', label='left', closed='left').ohlc()
    weekly_data.dropna(inplace=True)
    weekly_data.index = weekly_data.index.date
    weekly_data = weekly_data[~weekly_data.index.duplicated(keep='first')]
    weekly_data['weekly_return_pct'] = (weekly_data['close'] - weekly_data['open']) / weekly_data['open'] * 100
    weekly_data['avg_win_pct'] = weekly_data['weekly_return_pct'].apply(lambda x: x if x > 0 else 0).rolling(window=52, min_periods=1).mean()
    weekly_data['avg_loss_pct'] = weekly_data['weekly_return_pct'].apply(lambda x: abs(x) if x < 0 else 0).rolling(window=52, min_periods=1).mean()
    weekly_data['profit_factor'] = weekly_data.apply(
        lambda row: row['avg_win_pct'] / row['avg_loss_pct'] if row['avg_loss_pct'] > 0 else None,
        axis=1
    )
    rolling_std = weekly_data['weekly_return_pct'].rolling(window=52, min_periods=1).std()
    weekly_data['sharpe_ratio'] = (weekly_data['weekly_return_pct'].rolling(window=52, min_periods=1).mean() / rolling_std.replace(0, np.nan)) * np.sqrt(52)
    weekly_data.reset_index(inplace=True)
    weekly_data.rename(columns={'index': 'week_start_date'}, inplace=True)
    weekly_data = weekly_data.where(pd.notnull(weekly_data), None)
    return weekly_data.to_dict('records')

def store_results(symbol, weekly_stats):
    table_name = 'weekly_stock_performance'
    records_to_upsert = []
    for week in weekly_stats:
        records_to_upsert.append({
            'symbol': symbol,
            'week_start_date': week['week_start_date'].isoformat(),
            'open': week.get('open'),
            'high': week.get('high'),
            'low': week.get('low'),
            'close': week.get('close'),
            'weekly_return_pct': week.get('weekly_return_pct'),
            'avg_win_pct': week.get('avg_win_pct'),
            'avg_loss_pct': week.get('avg_loss_pct'),
            'profit_factor': week.get('profit_factor'),
            'sharpe_ratio': week.get('sharpe_ratio'),
            'last_updated': pytz.utc.localize(datetime.utcnow()).isoformat()
        })
    response = supabase.table(table_name).upsert(records_to_upsert, on_conflict='symbol,week_start_date').execute()
    print(f"Upsert response for {symbol}: {response}")

# --- API Routes ---

@app.route('/')
def home():
    return "Stock Analysis API is running!"

# +++++ NEW DIAGNOSTIC ROUTE +++++
@app.route('/test')
def test_route():
    """A simple test route to verify the app is running and can return JSON."""
    print("--- /test route was successfully hit! ---")
    return jsonify({"status": "ok", "message": "The Flask server is running correctly."})
# ++++++++++++++++++++++++++++++++

@app.route('/api/analyze/<symbol>', methods=['GET'])
def analyze_route(symbol):
    symbol = symbol.upper()
    if not symbol:
        return jsonify({"error": "Stock symbol is required"}), 400
    try:
        print(f"--- Starting analysis for {symbol} ---")
        stock_data = get_stock_data(symbol)
        if stock_data.empty:
            return jsonify({"error": f"Could not retrieve historical data for symbol: {symbol}"}), 404
        print(f"Step 1/3: Fetched data for {symbol}.")
        weekly_stats = process_data(stock_data)
        print(f"Step 2/3: Processed data for {symbol}.")
        store_results(symbol, weekly_stats)
        print(f"Step 3/3: Stored results for {symbol} in Supabase.")
        print(f"--- Analysis complete for {symbol} ---")
        return jsonify(weekly_stats)
    except Exception as e:
        print(f"CRITICAL ERROR in /analyze for symbol {symbol}:")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True)