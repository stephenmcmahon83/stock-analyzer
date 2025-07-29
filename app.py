from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import os

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
            week_info = {
                'week_start_date': week_start.strftime('%Y-%m-%d'),
                'week_return': float((week_group['Close'].iloc[-1] / week_group['Close'].iloc[0] - 1) * 100),
                'volume': float(week_group['Volume'].sum()),
                'volatility': float(week_group['Close'].pct_change().std() * np.sqrt(252) * 100) if len(week_group) > 1 else 0.0,
                'close_price': float(week_group['Close'].iloc[-1]),
                'high': float(week_group['High'].max()),
                'low': float(week_group['Low'].min()),
                'open_price': float(week_group['Open'].iloc[0])
            }
            for key, value in week_info.items():
                if key != 'week_start_date' and (np.isnan(value) or np.isinf(value)):
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
        
        filtered_data = weekly_data
        if filter_type == 'positive' or filter_type == 'after_up':
            filtered_data = [w for w in weekly_data if w['week_return'] > 0]
        elif filter_type == 'negative' or filter_type == 'after_down':
            filtered_data = [w for w in weekly_data if w['week_return'] < 0]
        
        returns = [w['week_return'] for w in filtered_data]
        if not returns:
            return jsonify({
                'weekly_data': [],
                'statistics': {
                    'total_weeks': 0, 'positive_weeks': 0, 'negative_weeks': 0,
                    'average_return': 0, 'best_week': 0, 'worst_week': 0,
                    'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0
                },
                'current_info': {'current_week': 'N/A', 'prior_week_return': 0},
                'history': []
            })
        
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]
        statistics = {
            'total_weeks': len(filtered_data),
            'positive_weeks': len(positive_returns),
            'negative_weeks': len(negative_returns),
            'average_return': float(np.mean(returns)) if returns else 0,
            'best_week': float(max(returns)) if returns else 0,
            'worst_week': float(min(returns)) if returns else 0,
            'win_rate': (len(positive_returns) / len(returns) * 100) if returns else 0,
            'profit_factor': calculate_profit_factor(positive_returns, negative_returns),
            'sharpe_ratio': calculate_sharpe_ratio(returns)
        }
        
        current_info = {
            'current_week': weekly_data[-1]['week_start_date'] if weekly_data else 'N/A',
            'prior_week_return': weekly_data[-2]['week_return'] if len(weekly_data) > 1 else 0
        }
        
        history = []
        for idx, week in enumerate(weekly_data[-20:]):
            week_date = pd.Timestamp(week['week_start_date'])
            history.append({
                'week_number': f"W{idx+1}",
                'year': week_date.year,
                'open_price': week['open_price'],
                'high_price': week['high'],
                'low_price': week['low'],
                'close_price': week['close_price'],
                'weekly_return_pct': week['week_return']
            })
        
        result = {
            'weekly_data': filtered_data[-52:],
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