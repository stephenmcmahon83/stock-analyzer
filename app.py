# app.py
import os
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from apscheduler.schedulers.background import BackgroundScheduler

# --- INITIALIZATION ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# Supabase client setup
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# --- FINANCIAL CALCULATIONS ---
def calculate_profit_factor(returns):
    """Calculates the ratio of gross profits to gross losses."""
    profits = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    if losses == 0:
        return float('inf') if profits > 0 else 0
    return profits / losses

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculates the annualized Sharpe Ratio for weekly returns."""
    if not returns or len(returns) < 2 or np.std(returns) == 0:
        return 0
    # Convert from % to decimal for calculation
    returns_decimal = [r / 100 for r in returns]
    avg_return = np.mean(returns_decimal)
    std_dev = np.std(returns_decimal)
    
    # Annualize the weekly figures (52 weeks in a year)
    annualized_return = avg_return * 52
    annualized_std_dev = std_dev * np.sqrt(52)
    if annualized_std_dev == 0:
        return 0
    return (annualized_return - risk_free_rate) / annualized_std_dev

# --- DATA FETCHING & PROCESSING ---
def fetch_and_store_stock_data(symbol: str):
    """Fetches historical data from yfinance, processes it into weekly format, and stores it in Supabase."""
    try:
        symbol = symbol.upper()
        # 1. Get or create the stock record in the 'stocks' table
        stock_res = supabase.table('stocks').upsert({'symbol': symbol}, on_conflict='symbol').execute()
        stock_id = stock_res.data[0]['id']

        # 2. Fetch up to 20 years of data from yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="20y", interval="1d")
        if df.empty:
            return False

        # 3. Resample daily data to weekly data (Monday-based week)
        df.index = pd.to_datetime(df.index)
        weekly_df = df.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()

        # 4. Calculate required fields
        weekly_df['year'] = weekly_df.index.year
        weekly_df['week_number'] = weekly_df.index.isocalendar().week
        weekly_df['weekly_return_pct'] = ((weekly_df['Close'] - weekly_df['Open']) / weekly_df['Open']) * 100

        # 5. Prepare data for Supabase insertion
        records_to_insert = []
        for index, row in weekly_df.iterrows():
            records_to_insert.append({
                'stock_id': stock_id,
                'year': int(row['year']),
                'week_number': int(row['week_number']),
                'open_price': float(row['Open']),
                'high_price': float(row['High']),
                'low_price': float(row['Low']),
                'close_price': float(row['Close']),
                'weekly_return_pct': float(row['weekly_return_pct'])
            })

        # 6. Upsert data into Supabase (insert new, update existing)
        if records_to_insert:
            supabase.table('weekly_prices').upsert(records_to_insert, on_conflict='stock_id, year, week_number').execute()

        # 7. Update the 'last_updated' timestamp for the stock
        supabase.table('stocks').update({'last_updated': datetime.now(pytz.UTC).isoformat()}).eq('symbol', symbol).execute()
        return True
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return False

# --- FLASK API ENDPOINTS ---
@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/api/analyze/<symbol>')
def analyze_stock(symbol):
    """The main API endpoint to get all analysis data for a stock."""
    try:
        symbol = symbol.upper()
        # 1. Check when the stock was last updated
        stock_res = supabase.table('stocks').select('last_updated').eq('symbol', symbol).execute()
        needs_update = True
        if stock_res.data and stock_res.data[0].get('last_updated'):
            last_updated_str = stock_res.data[0]['last_updated']
            last_updated = datetime.fromisoformat(last_updated_str)
            # Only update if data is older than 24 hours
            if datetime.now(pytz.UTC) - last_updated < timedelta(days=1):
                needs_update = False
        
        if needs_update:
            if not fetch_and_store_stock_data(symbol):
                 return jsonify({'error': f'Could not retrieve data for symbol: {symbol}'}), 404

        # 2. Fetch all historical weekly data for the stock
        stock_id_res = supabase.table('stocks').select('id').eq('symbol', symbol).execute()
        if not stock_id_res.data:
            return jsonify({'error': 'Stock ID not found after fetch.'}), 404
        stock_id = stock_id_res.data[0]['id']

        price_res = supabase.table('weekly_prices').select('year, week_number, open_price, high_price, low_price, close_price, weekly_return_pct') \
            .eq('stock_id', stock_id) \
            .order('year', desc=True) \
            .order('week_number', desc=True) \
            .execute()
        
        if not price_res.data:
            return jsonify({'error': 'No weekly data found after fetch.'}), 404

        df = pd.DataFrame(price_res.data)
        
        # 3. Create the historical data table for the current year
        current_year = datetime.now().year
        current_year_data = df[df['year'] == current_year].sort_values('week_number').to_dict('records')

        # 4. Create the statistics table
        filter_type = request.args.get('filter', 'all') # 'all', 'after_up', 'after_down'
        
        df_sorted = df.sort_values(['year', 'week_number'], ascending=True)
        df_sorted['prev_week_return'] = df_sorted['weekly_return_pct'].shift(1)
        
        filtered_df = df_sorted.copy()
        if filter_type == 'after_up':
            filtered_df = df_sorted[df_sorted['prev_week_return'] > 0]
        elif filter_type == 'after_down':
            filtered_df = df_sorted[df_sorted['prev_week_return'] < 0]
            
        # Group data and calculate stats
        stats_groups = filtered_df.groupby('week_number')['weekly_return_pct']
        
        stats_list = []
        for name, group in stats_groups:
            stats_list.append({
                'week_number': name,
                'count': group.count(),
                'avg_return': group.mean(),
                'std_dev': group.std(),
                'pct_profitable': (group > 0).sum() / group.count() * 100 if group.count() > 0 else 0,
                'profit_factor': calculate_profit_factor(group.tolist()),
                'sharpe_ratio': calculate_sharpe_ratio(group.tolist())
            })
        
        stats_by_week = pd.DataFrame(stats_list)

        # 5. Get current week info
        today = datetime.now()
        current_week_info = {
            'current_week': today.isocalendar()[1],
            'prior_week_return': df.iloc[0]['weekly_return_pct'] if not df.empty else 0
        }
        
        return jsonify({
            'statistics': stats_by_week.to_dict('records'),
            'history': current_year_data,
            'current_info': current_week_info
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- AUTOMATION SCHEDULER ---
def scheduled_update_job():
    """Job to update all stocks in the database."""
    print(f"Running scheduled update at {datetime.now()}...")
    with app.app_context():
        try:
            stocks_res = supabase.table('stocks').select('symbol').execute()
            if stocks_res.data:
                for stock in stocks_res.data:
                    print(f"Updating {stock['symbol']}...")
                    fetch_and_store_stock_data(stock['symbol'])
            print("Scheduled update finished.")
        except Exception as e:
            print(f"Error during scheduled update: {e}")

# Run the update job every day at 10:00 PM UTC (6 PM EST, after US markets close)
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(scheduled_update_job, 'cron', hour=22, minute=0, timezone='UTC')
scheduler.start()

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)