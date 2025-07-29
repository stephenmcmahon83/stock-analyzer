from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests from GitHub Pages

def fetch_stock_data(symbol, period="30y"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_weekly_ohlc(data):
    weekly_data = []
    # Resample daily data to weekly data (Monday start)
    weekly = data.resample('W-MON')
    
    for week_start, week_group in weekly:
        if len(week_group) == 0:
            continue
            
        try:
            week_date = pd.Timestamp(week_start)
            week_info = {
                'week_start_date': week_start.strftime('%Y-%m-%d'),
                'week_number': week_date.isocalendar()[1],  # ISO week number (1-53)
                'year': week_date.year,
                'open': float(week_group['Open'].iloc[0]),
                'high': float(week_group['High'].max()),
                'low': float(week_group['Low'].min()),
                'close': float(week_group['Close'].iloc[-1]),
                'change_pct': float((week_group['Close'].iloc[-1] / week_group['Open'].iloc[0] - 1) * 100)
            }
            weekly_data.append(week_info)
        except Exception as e:
            print(f"Error calculating weekly OHLC for {week_start}: {e}")
            continue
    
    return weekly_data

@app.route('/')
def home():
    return jsonify({"message": "Stock OHLC API is running"})

@app.route('/api/ohlc/<symbol>')
def get_ohlc_data(symbol):
    try:
        print(f"Fetching weekly OHLC data for {symbol}")
        stock_data = fetch_stock_data(symbol)
        
        if stock_data is None or stock_data.empty:
            return jsonify({"error": f"No data found for symbol {symbol}"}), 404
        
        weekly_ohlc = calculate_weekly_ohlc(stock_data)
        
        if not weekly_ohlc:
            return jsonify({"error": "No weekly data could be calculated"}), 500
        
        result = {
            'weekly_ohlc': weekly_ohlc
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return jsonify({"error": "An unexpected server error occurred"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)