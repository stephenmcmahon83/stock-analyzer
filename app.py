from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

def fetch_stock_data(symbol, period="5y"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_weekly_metrics(data):
    weekly_data = []
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
                'week_return': float((week_group['Close'].iloc[-1] / week_group['Close'].iloc[0] - 1) * 100),
                'volume': float(week_group['Volume'].sum()),
                'volatility': float(week_group['Close'].pct_change().std() * np.sqrt(252) * 100) if len(week_group) > 1 else 0.0,
                'close_price': float(week_group['Close'].iloc[-1]),
                'high': float(week_group['High'].max()),
                'low': float(week_group['Low'].min()),
                'open_price': float(week_group['Open'].iloc[0])
            }
            for key, value in week_info.items():
                if key != 'week_start_date' and key != 'week_number' and key != 'year' and (np.isnan(value) or np.isinf(value)):
                    week_info[key] = 0.0
            weekly_data.append(week_info)
        except Exception as e:
            print(f"Error calculating metrics for week {week_start}: {e}")
            continue
    
    return weekly_data

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    else:
        return obj

def calculate_profit_factor(positive_returns, negative_returns):
    if not negative_returns:
        return 10.0 if positive_returns else 0.0
    total_gains = sum(positive_returns)
    total_losses = abs(sum(negative_returns))
    if total_losses == 0:
        return 10.0 if total_gains > 0 else 0.0
    return float(total_gains / total_losses)

def calculate_sharpe_ratio(returns):
    if not returns or len(returns) < 2:
        return 0.0
    returns_array = np.array(returns) / 100
    avg_return = np.mean(returns_array)
    std_dev = np.std(returns_array, ddof=1)
    if std_dev == 0 or np.isnan(std_dev):
        return 0.0
    sharpe = (avg_return / std_dev) * np.sqrt(52)
    if np.isnan(sharpe) or np.isinf(sharpe):
        return 0.0
    return float(sharpe)

@app.route('/')
def home():
    return jsonify({"message": "Stock Analysis API is running"})

@app.route('/api/analyze/<symbol>')
def analyze_route(symbol):
    try:
        print(f"--- Starting analysis for {symbol} with filter: {request.args.get('filter', 'all')} ---")
        filter_type = request.args.get('filter', 'all')
        stock_data = fetch_stock_data(symbol.upper())
        
        if stock_data is None or stock_data.empty:
            return jsonify({"error": f"No data found for symbol {symbol}"}), 404
        
        print(f"Fetched {len(stock_data)} days of data for {symbol}")
        weekly_data = calculate_weekly_metrics(stock_data)
        
        if not weekly_data:
            return jsonify({"error": "No weekly data could be calculated"}), 500
        
        # Determine current week number based on the latest week in data or current date
        current_week_num = weekly_data[-1]['week_number'] if weekly_data else datetime.now().isocalendar()[1]
        
        # Apply filters based on week number and prior week performance
        filtered_data = weekly_data
        current_week_data = [w for w in weekly_data if w['week_number'] == current_week_num]
        if filter_type == 'after_up':
            # Get Week N performance following an up week in Week N-1
            filtered_data = []
            for i in range(1, len(weekly_data)):
                prev_week = weekly_data[i-1]
                curr_week = weekly_data[i]
                if curr_week['week_number'] == current_week_num and prev_week['week_number'] == current_week_num - 1 and prev_week['week_return'] > 0:
                    filtered_data.append(curr_week)
        elif filter_type == 'after_down':
            # Get Week N performance following a down week in Week N-1
            filtered_data = []
            for i in range(1, len(weekly_data)):
                prev_week = weekly_data[i-1]
                curr_week = weekly_data[i]
                if curr_week['week_number'] == current_week_num and prev_week['week_number'] == current_week_num - 1 and prev_week['week_return'] < 0:
                    filtered_data.append(curr_week)
        elif filter_type == 'all':
            # Only current week number across all years
            filtered_data = current_week_data
        
        # Calculate statistics for the filtered data (historical performance for current week number)
        returns = [w['week_return'] for w in filtered_data]
        statistics = []
        if returns:
            positive_returns = [r for r in returns if r > 0]
            negative_returns = [r for r in returns if r < 0]
            statistics = [{
                'week_number': str(current_week_num),
                'count': len(filtered_data),
                'avg_return': float(np.mean(returns)) if returns else 0,
                'profitable_percent': (len(positive_returns) / len(returns) * 100) if returns else 0,
                'profit_factor': calculate_profit_factor(positive_returns, negative_returns),
                'sharpe_ratio': calculate_sharpe_ratio(returns),
                'std_dev': float(np.std(returns)) if len(returns) > 1 else 0
            }]
        else:
            statistics = [{
                'week_number': str(current_week_num),
                'count': 0,
                'avg_return': 0,
                'profitable_percent': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'std_dev': 0
            }]
        
        # Info for the current week (latest occurrence of the week number)
        current_info = {
            'current_week': str(current_week_num),
            'prior_week_return': weekly_data[-2]['week_return'] if len(weekly_data) > 1 and weekly_data[-1]['week_number'] == current_week_num else 0
        }
        
        # History table shows ALL historical data for all week numbers across years
        history = []
        for week in weekly_data:
            history.append({
                'week_number': str(week['week_number']),
                'year': week['year'],
                'open': week['open_price'],
                'high': week['high'],
                'low': week['low'],
                'close': week['close_price'],
                'return': week['week_return']
            })
        
        result = {
            'weekly_data': filtered_data,
            'statistics': statistics,
            'current_info': current_info,
            'history': history
        }
        
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        print(f"ERROR analyzing {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)