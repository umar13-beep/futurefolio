# FutureFolio - Complete Backend with Login (Railway Ready)
from flask import Flask, jsonify, request, send_file, send_from_directory, abort
from flask_cors import CORS
import yfinance as yf
from datetime import datetime
import json
import os
import sys
import io
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import requests

# ========== RAILWAY CONFIGURATION ==========
# Get environment variables for Railway
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'False') == 'True'

# ========== PATH CONFIGURATION ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Smart frontend directory detection
# Option 1: Frontend folder in same directory as app.py
FRONTEND_DIR_SAME = os.path.join(BASE_DIR, "frontend")

# Option 2: Frontend folder one level up (original structure)
FRONTEND_DIR_PARENT = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

# Option 3: Frontend folder in static directory (for deployment)
FRONTEND_DIR_STATIC = os.path.join(BASE_DIR, "static")

# Determine which frontend directory exists
if os.path.exists(FRONTEND_DIR_SAME):
    FRONTEND_DIR = FRONTEND_DIR_SAME
    print(f"✅ Using frontend from: {FRONTEND_DIR_SAME}")
elif os.path.exists(FRONTEND_DIR_PARENT):
    FRONTEND_DIR = FRONTEND_DIR_PARENT
    print(f"✅ Using frontend from: {FRONTEND_DIR_PARENT}")
elif os.path.exists(FRONTEND_DIR_STATIC):
    FRONTEND_DIR = FRONTEND_DIR_STATIC
    print(f"✅ Using frontend from: {FRONTEND_DIR_STATIC}")
else:
    # Create frontend directory if it doesn't exist
    FRONTEND_DIR = FRONTEND_DIR_SAME
    os.makedirs(FRONTEND_DIR, exist_ok=True)
    print(f"⚠️ Created frontend directory: {FRONTEND_DIR}")

# Database folder (always in same directory as app.py)
DB_FOLDER = os.path.join(BASE_DIR, "database")
os.makedirs(DB_FOLDER, exist_ok=True)

USERS_FILE = os.path.join(DB_FOLDER, "users.json")
PORTFOLIO_FILE = os.path.join(DB_FOLDER, "portfolio.json")
WISHLIST_FILE = os.path.join(DB_FOLDER, "wishlist.json")

# ========== FLASK APP INITIALIZATION ==========
app = Flask(__name__)

# Configure CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Add CORS headers after each request for production
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

print("=" * 50)
print("FUTUREFOLIO STARTING...")
print(f"Base Directory: {BASE_DIR}")
print(f"Frontend Directory: {FRONTEND_DIR}")
print(f"Database Directory: {DB_FOLDER}")
print(f"Debug Mode: {DEBUG}")
print("=" * 50)

# ========== DATABASE FUNCTIONS ==========
def init_database():
    """Initialize database files"""
    for file in [USERS_FILE, PORTFOLIO_FILE, WISHLIST_FILE]:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                json.dump({}, f)
                print(f"✅ Created database file: {file}")

def load_users():
    """Load users from file"""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    """Save users to file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_wishlist():
    """Load wishlist data from file."""
    try:
        with open(WISHLIST_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_wishlist(wishlist):
    """Save wishlist data to file."""
    with open(WISHLIST_FILE, 'w') as f:
        json.dump(wishlist, f, indent=2)

def load_portfolio_data():
    """Load portfolio data from file."""
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_portfolio_data(portfolio):
    """Save portfolio data to file."""
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)

# Initialize database
init_database()

# ========== FRONTEND ROUTING ==========
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve frontend files"""
    # Don't interfere with API routes
    if path.startswith('api') or path.startswith('static'):
        abort(404)
    
    # Serve index.html for root path
    if path == '' or path == '/':
        index_path = os.path.join(FRONTEND_DIR, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(FRONTEND_DIR, 'index.html')
        else:
            return jsonify({"error": "Frontend not found. Please check your folder structure."}), 404
    
    # Check if file exists directly
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.isfile(file_path):
        return send_from_directory(FRONTEND_DIR, path)
    
    # Check for HTML file without extension (SPA routes)
    if '.' not in path:
        html_path = os.path.join(FRONTEND_DIR, f"{path}.html")
        if os.path.isfile(html_path):
            return send_from_directory(FRONTEND_DIR, f"{path}.html")
    
    # Fallback to index.html for SPA routing
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return send_from_directory(FRONTEND_DIR, "index.html")
    
    return jsonify({"error": "Page not found"}), 404

# ========== API STATUS ==========
@app.route('/api/status')
def api_status():
    return jsonify({
        "app": "FutureFolio",
        "version": "2.0.0",
        "status": "Running on Railway",
        "frontend_dir": FRONTEND_DIR,
        "database_dir": DB_FOLDER,
        "message": "AI Stock Prediction System",
        "endpoints": {
            "/api/register": "POST - Register user",
            "/api/login": "POST - User login",
            "/api/stocks": "GET - Get stock list",
            "/api/stock/<symbol>": "GET - Get stock data",
            "/api/predict/<symbol>": "GET - AI stock prediction",
            "/api/news": "GET - Market news",
            "/api/portfolio/<email>": "GET - User portfolio",
            "/api/portfolio/add": "POST - Add to portfolio",
            "/api/portfolio/sell": "POST - Sell from portfolio",
            "/api/wishlist": "GET/POST - Wishlist operations",
            "/api/chat": "POST - Chatbot assistant",
            "/api/report/pdf": "POST - Generate PDF report"
        }
    })

# ========== AUTHENTICATION API ==========
@app.route('/api/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        name = data.get('name', '')
        
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'}), 400
        
        users = load_users()
        
        if email in users:
            return jsonify({'success': False, 'error': 'User already exists'}), 400
        
        users[email] = {
            'email': email,
            'password': password,  # Note: In real app, hash passwords!
            'name': name,
            'created_at': datetime.now().isoformat(),
            'balance': 100000  # Starting balance
        }
        
        save_users(users)
        return jsonify({'success': True, 'message': 'User created successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'}), 400
        
        users = load_users()
        
        if email not in users:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        if users[email]['password'] != password:
            return jsonify({'success': False, 'error': 'Invalid password'}), 401
        
        # Don't send password back
        user_data = users[email].copy()
        user_data.pop('password', None)
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': user_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== STOCK API ==========
@app.route('/api/stocks')
def get_stocks():
    """Get popular Indian stocks"""
    stocks = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Conglomerate"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT"},
        {"symbol": "INFY.NS", "name": "Infosys", "sector": "IT"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Banking"},
        {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom"},
        {"symbol": "ITC.NS", "name": "ITC Limited", "sector": "FMCG"},
        {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking"}
    ]
    return jsonify({"stocks": stocks, "count": len(stocks)})

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    """Get current stock price"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d")
        
        if hist.empty:
            return jsonify({"error": "No data found"}), 404
            
        data = {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "current_price": round(hist['Close'].iloc[-1], 2),
            "open": round(hist['Open'].iloc[-1], 2),
            "high": round(hist['High'].iloc[-1], 2),
            "low": round(hist['Low'].iloc[-1], 2),
            "volume": int(hist['Volume'].iloc[-1]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== ML HELPERS ==========
_PREDICT_CACHE = {}

def _build_supervised_lags(series: np.ndarray, lookback: int = 30, horizon: int = 1):
    """
    Convert a 1D time-series into supervised learning samples using lag windows.
    X[t] = series[t-lookback : t]
    y[t] = series[t + horizon]
    """
    if len(series) <= lookback + horizon + 5:
        raise ValueError("Insufficient time-series length for the selected lookback.")

    X, y = [], []
    for i in range(lookback, len(series) - horizon):
        X.append(series[i - lookback : i])
        y.append(series[i + horizon - 1])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

def _time_series_predict_regression(symbol: str, lookback: int = 30, horizon: int = 1):
    """
    Forecast next-day closing price using regression on lag windows.
    Trains two candidate regressors and selects the best by time-based validation RMSE.
    """
    cache_key = f"{symbol}|{lookback}|{horizon}"
    if cache_key in _PREDICT_CACHE:
        return _PREDICT_CACHE[cache_key]

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y", interval="1d")
    if hist is None or hist.empty:
        raise ValueError("No historical data found for the provided symbol.")
    if "Close" not in hist.columns:
        raise ValueError("Historical data is missing 'Close' column.")

    hist = hist.dropna(subset=["Close"]).sort_index()
    closes = hist["Close"].astype(float).values

    X, y = _build_supervised_lags(closes, lookback=lookback, horizon=horizon)

    split_idx = int(len(X) * 0.8)
    if split_idx < 10:
        raise ValueError("Not enough samples to train/validate the model.")

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    candidates = [
        ("linear_regression", make_pipeline(StandardScaler(), LinearRegression())),
        ("random_forest", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )),
    ]

    best_name = None
    best_model = None
    best_rmse = None
    for name, model in candidates:
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    latest_window = closes[-lookback:]
    next_close = float(best_model.predict(latest_window.reshape(1, -1))[0])
    current_close = float(closes[-1])

    # Confidence heuristic: scale validation RMSE relative to current close.
    if current_close > 0:
        rel_err = best_rmse / current_close
        confidence = float(np.clip(100 * (1 - rel_err), 10, 95))
    else:
        confidence = 50.0

    trend = "up" if next_close >= current_close else "down"

    # Return compact history for charting
    plot_days = min(60, len(hist))
    plot_hist = hist.tail(plot_days)
    history_dates = [d.strftime("%Y-%m-%d") for d in plot_hist.index.to_pydatetime()]
    history_closes = [float(v) for v in plot_hist["Close"].values]

    result = {
        "symbol": symbol,
        "current_price": round(current_close, 2),
        "predicted_price": round(next_close, 2),
        "confidence": f"{round(confidence)}%",
        "trend": trend,
        "message": f"Regression forecast from last 1y data (model: {best_name}).",
        "model_type": best_name,
        "history": {
            "dates": history_dates,
            "closes": history_closes,
        },
    }

    _PREDICT_CACHE[cache_key] = result
    return result

@app.route('/api/predict/<symbol>')
def predict_stock(symbol):
    """AI prediction endpoint (regression time-series forecast)."""
    try:
        result = _time_series_predict_regression(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== WISHLIST API ==========
@app.route('/api/wishlist/add', methods=['POST'])
def wishlist_add():
    """Add a stock to the user's wishlist."""
    try:
        data = request.json or {}
        email = data.get('email')
        symbol = data.get('symbol')
        name = data.get('name', symbol)

        if not email or not symbol:
            return jsonify({'success': False, 'error': 'Email and symbol required'}), 400

        wishlist = load_wishlist()
        if email not in wishlist:
            wishlist[email] = []

        # Avoid duplicates by symbol
        existing = next((x for x in wishlist[email] if x.get('symbol') == symbol), None)
        if existing:
            existing['name'] = name or existing.get('name') or symbol
            existing['updated_at'] = datetime.now().isoformat()
        else:
            wishlist[email].append({
                'symbol': symbol,
                'name': name or symbol,
                'added_at': datetime.now().isoformat(),
            })

        save_wishlist(wishlist)
        return jsonify({'success': True, 'message': 'Added to wishlist'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/wishlist/<email>')
def wishlist_get(email):
    """Get user's wishlist."""
    try:
        wishlist = load_wishlist()
        items = wishlist.get(email, [])
        return jsonify({'success': True, 'wishlist': items, 'count': len(items)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/wishlist/remove', methods=['POST'])
def wishlist_remove():
    """Remove a stock from wishlist."""
    try:
        data = request.json or {}
        email = data.get('email')
        symbol = data.get('symbol')

        if not email or not symbol:
            return jsonify({'success': False, 'error': 'Email and symbol required'}), 400

        wishlist = load_wishlist()
        items = wishlist.get(email, [])
        wishlist[email] = [x for x in items if x.get('symbol') != symbol]
        save_wishlist(wishlist)
        return jsonify({'success': True, 'message': 'Removed from wishlist'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== RECOMMENDATION API ==========
@app.route('/api/recommend/<symbol>')
def recommend_stock(symbol):
    """Generate a simple recommendation based on forecast trend and confidence."""
    try:
        prediction = _time_series_predict_regression(symbol)
        confidence_num = float(str(prediction.get('confidence', '0%')).replace('%', '').strip() or 0)
        trend = prediction.get('trend')

        # Simple rule-based recommendation
        if trend == 'up':
            action = 'Buy' if confidence_num >= 60 else 'Watch'
        elif trend == 'down':
            action = 'Sell' if confidence_num >= 60 else 'Hold'
        else:
            action = 'Hold'

        if confidence_num >= 75:
            risk = 'Low'
        elif confidence_num >= 50:
            risk = 'Medium'
        else:
            risk = 'High'

        rationale = (
            f"Forecast indicates a '{trend}' movement with confidence {prediction.get('confidence')}. "
            f"Recommendation '{action}' is based on this expected direction and model reliability estimate."
        )

        return jsonify({
            'symbol': symbol,
            'action': action,
            'risk': risk,
            'confidence': prediction.get('confidence'),
            'trend': trend,
            'rationale': rationale,
            'predicted_price': prediction.get('predicted_price'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== CHATBOT API ==========
@app.route('/api/chat', methods=['POST'])
def chat():
    """Simple rule-based chatbot"""
    try:
        data = request.json or {}
        message = (data.get('message') or '').strip()
        email = data.get('email')

        if not message:
            return jsonify({'reply': 'Please send a message. Try "help" or "predict RELIANCE.NS".'})

        msg = message.lower()

        def extract_symbol(text: str):
            text_up = text.upper()
            m = re.findall(r'([A-Z0-9][A-Z0-9\\.\\-]*\\.(?:NS|BO|BSE|NSE))', text_up)
            if m:
                return m[0]
            tokens = text_up.split()
            for t in reversed(tokens):
                if '.' in t and len(t) <= 25:
                    return t
            return None

        if 'help' in msg:
            return jsonify({
                'reply': (
                    'I can help with predictions and current prices. Examples: '
                    '"predict RELIANCE.NS", "price TCS.NS", or ask "help".'
                )
            })

        symbol = extract_symbol(message)
        if 'predict' in msg or 'forecast' in msg:
            if not symbol:
                return jsonify({'reply': 'Please specify a symbol. Example: "predict RELIANCE.NS".'})
            prediction = _time_series_predict_regression(symbol)

            confidence_num = float(str(prediction.get('confidence', '0%')).replace('%', '').strip() or 0)
            trend = prediction.get('trend')
            if trend == 'up':
                action = 'Buy' if confidence_num >= 60 else 'Watch'
            elif trend == 'down':
                action = 'Sell' if confidence_num >= 60 else 'Hold'
            else:
                action = 'Hold'

            reply = (
                f"Prediction for {symbol}: predicted price {prediction.get('predicted_price')} "
                f"with a '{prediction.get('trend')}' trend and confidence {prediction.get('confidence')}. "
            )
            if action:
                reply += f"Recommendation: {action}. "
            reply += f"Message: {prediction.get('message')}"
            return jsonify({'reply': reply})

        if 'price' in msg or 'current price' in msg:
            if not symbol:
                return jsonify({'reply': 'Please specify a symbol. Example: "price HDFCBANK.NS".'})
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            if hist is None or hist.empty:
                return jsonify({'reply': f'No data found for {symbol}.'})

            current_price = round(float(hist['Close'].iloc[-1]), 2)
            reply = f"Current price for {symbol} is approximately INR {current_price}."
            return jsonify({'reply': reply})

        return jsonify({
            'reply': (
                'I did not understand the request. Try: '
                '"help", "predict RELIANCE.NS", or "price TCS.NS".'
            )
        })
    except Exception as e:
        return jsonify({'reply': f'Chatbot error: {str(e)}'})

# ========== PDF REPORT API ==========
@app.route('/api/report/pdf', methods=['POST'])
def pdf_report():
    """Generate a simple PDF summary for the user's portfolio."""
    try:
        data = request.json or {}
        email = data.get('email')
        if not email:
            return jsonify({'success': False, 'error': 'Email is required'}), 400

        portfolio_store = load_portfolio_data()
        items = portfolio_store.get(email, [])
        total_invested = sum(item.get('total_invested', 0) for item in items)

        # Estimate current value using yfinance
        total_current_value = 0.0
        priced_items = []
        for item in items:
            symbol = item.get('symbol')
            quantity = int(item.get('quantity', 0))
            buy_price = float(item.get('buy_price', 0))
            current_price = None
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1d")
                if hist is not None and not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
            except:
                current_price = None

            if current_price is None:
                current_price = buy_price * 1.05

            current_value = current_price * quantity
            total_current_value += current_value
            priced_items.append({
                **item,
                'current_price': round(current_price, 2),
                'current_value': round(current_value, 2),
            })

        profit_loss = total_current_value - total_invested
        roi = (profit_loss / total_invested * 100) if total_invested > 0 else 0.0

        # Create PDF in memory
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter

        y = height - 60
        c.setFont("Helvetica-Bold", 16)
        c.drawString(60, y, "FutureFolio - Portfolio Report")

        c.setFont("Helvetica", 11)
        y -= 22
        c.drawString(60, y, f"User: {email}")

        y -= 18
        c.drawString(60, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        y -= 28
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y, "Portfolio Summary")

        y -= 18
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"Total Invested: INR {total_invested:.2f}")
        y -= 14
        c.drawString(60, y, f"Current Value: INR {total_current_value:.2f}")
        y -= 14
        c.drawString(60, y, f"Profit/Loss: INR {profit_loss:.2f}")
        y -= 14
        c.drawString(60, y, f"ROI: {roi:.2f}%")

        y -= 24
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y, "Holdings")

        y -= 18
        c.setFont("Helvetica", 10)

        for item in priced_items[:18]:
            if y < 60:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 10)
            symbol = item.get('symbol', '')
            quantity = item.get('quantity', 0)
            buy_price = float(item.get('buy_price', 0))
            current_price = item.get('current_price', 0)
            current_value = item.get('current_value', 0)
            total_invested_item = item.get('total_invested', 0)
            pl_item = current_value - total_invested_item

            text = f"{symbol} | Qty: {quantity} | Buy: {buy_price:.2f} | Current: {current_price:.2f} | P/L: {pl_item:.2f}"
            c.drawString(60, y, text[:130])
            y -= 14

        if not priced_items:
            c.drawString(60, y, "No holdings found in this portfolio.")

        c.showPage()
        c.save()
        buf.seek(0)

        filename = f"FutureFolio_Report_{email.replace('@', '_at_').replace('.', '_')}.pdf"
        return send_file(
            buf,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== PORTFOLIO API ==========
@app.route('/api/portfolio/add', methods=['POST'])
def add_portfolio():
    """Add stock to portfolio"""
    try:
        data = request.json
        email = data.get('email')
        symbol = data.get('symbol')
        name = data.get('name')
        quantity = data.get('quantity')
        price = data.get('price')
        
        if not all([email, symbol, name, quantity, price]):
            return jsonify({'success': False, 'error': 'All fields required'}), 400
        
        portfolio = load_portfolio_data()
        
        if email not in portfolio:
            portfolio[email] = []
        
        portfolio[email].append({
            'symbol': symbol,
            'name': name,
            'quantity': int(quantity),
            'buy_price': float(price),
            'total_invested': int(quantity) * float(price),
            'date': datetime.now().isoformat(),
            'transaction_id': len(portfolio[email]) + 1
        })
        
        save_portfolio_data(portfolio)
        
        return jsonify({'success': True, 'message': 'Stock added to portfolio'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio/<email>')
def get_user_portfolio(email):
    """Get user portfolio"""
    try:
        portfolio = load_portfolio_data()
        user_portfolio = portfolio.get(email, [])
        
        # Calculate summary with real-time prices
        total_invested = sum(item['total_invested'] for item in user_portfolio)
        
        # Calculate current value using yfinance
        current_value = 0
        for item in user_portfolio:
            try:
                stock = yf.Ticker(item['symbol'])
                hist = stock.history(period="1d")
                if hist is not None and not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                else:
                    current_price = item['buy_price'] * 1.05
            except:
                current_price = item['buy_price'] * 1.05
            
            current_value += current_price * item['quantity']
        
        profit_loss = current_value - total_invested
        profit_percentage = (profit_loss / total_invested * 100) if total_invested > 0 else 0
        
        summary = {
            'total_invested': total_invested,
            'current_value': current_value,
            'profit_loss': profit_loss,
            'profit_percentage': profit_percentage
        }
        
        return jsonify({
            'success': True,
            'portfolio': user_portfolio,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio/sell', methods=['POST'])
def sell_from_portfolio():
    """Sell shares from a specific portfolio transaction."""
    try:
        data = request.json or {}
        email = data.get('email')
        transaction_id = data.get('transaction_id')
        quantity = data.get('quantity')

        if not all([email, transaction_id, quantity]):
            return jsonify({'success': False, 'error': 'Email, transaction_id, and quantity required'}), 400

        quantity = int(quantity)
        transaction_id = int(transaction_id)

        if quantity <= 0:
            return jsonify({'success': False, 'error': 'Quantity must be positive'}), 400

        portfolio_store = load_portfolio_data()
        user_items = portfolio_store.get(email, [])
        
        if not user_items:
            return jsonify({'success': False, 'error': 'Portfolio is empty'}), 400

        updated_items = []
        sold = False
        for item in user_items:
            if int(item.get('transaction_id', -1)) == transaction_id:
                current_qty = int(item.get('quantity', 0))
                if quantity > current_qty:
                    return jsonify({'success': False, 'error': 'Sell quantity exceeds owned quantity'}), 400

                new_qty = current_qty - quantity
                buy_price = float(item.get('buy_price', 0))

                if new_qty > 0:
                    item['quantity'] = new_qty
                    item['total_invested'] = new_qty * buy_price
                    item['sold_at'] = datetime.now().isoformat()
                    updated_items.append(item)
                sold = True
            else:
                updated_items.append(item)

        if not sold:
            return jsonify({'success': False, 'error': 'Transaction not found'}), 404

        portfolio_store[email] = updated_items
        save_portfolio_data(portfolio_store)

        return jsonify({'success': True, 'message': 'Stock sold successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== NEWS API ==========
@app.route('/api/news', methods=['GET'])
def news():
    """Fetch market news."""
    try:
        q = (request.args.get('q') or '').strip()
        symbol = (request.args.get('symbol') or '').strip()
        api_key = os.getenv('NEWSAPI_KEY', '').strip()

        def _clean_symbol(sym: str):
            s = (sym or '').upper().strip()
            for suffix in ['.NS', '.BO', '.BSE', '.NSE']:
                if s.endswith(suffix):
                    s = s[: -len(suffix)]
                    break
            return s.strip()

        cleaned = _clean_symbol(symbol)
        query = q if q else (cleaned + ' stock' if cleaned else (symbol if symbol else 'stock market'))

        def _normalize_articles(items):
            normalized = []
            for a in items or []:
                normalized.append({
                    'source': (a.get('source') or {}).get('name') or a.get('source'),
                    'title': a.get('title') or '',
                    'publishedAt': a.get('publishedAt') or '',
                    'description': a.get('description') or '',
                    'url': a.get('url') or ''
                })
            return normalized

        if api_key:
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'sortBy': 'publishedAt',
                'pageSize': 6,
                'language': 'en',
                'apiKey': api_key,
            }
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('status') == 'ok':
                    return jsonify({'success': True, 'articles': _normalize_articles(data.get('articles', []))})

        # Fallback sample news
        fallback = [
            {
                'source': 'FutureFolio',
                'title': 'Market remains volatile as investors assess macro signals',
                'publishedAt': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'description': 'Analysts note that short-term price movements are influenced by global risk sentiment and sector rotation.',
                'url': ''
            },
            {
                'source': 'FutureFolio',
                'title': 'AI-driven strategies increase interest in systematic forecasting',
                'publishedAt': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'description': 'New research highlights that models trained on historical patterns can assist decision-making.',
                'url': ''
            },
            {
                'source': 'FutureFolio',
                'title': 'Investors focus on earnings and guidance for near-term direction',
                'publishedAt': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'description': 'Company updates and guidance often drive momentum and volatility.',
                'url': ''
            },
        ]

        return jsonify({'success': True, 'articles': fallback})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== MAIN ENTRY POINT ==========
if __name__ == '__main__':
    print("=" * 50)
    print("🚀 FUTUREFOLIO IS RUNNING")
    print(f"📍 URL: http://localhost:{PORT}")
    print(f"📁 Frontend: {FRONTEND_DIR}")
    print(f"💾 Database: {DB_FOLDER}")
    print(f"🐍 Python Version: {sys.version}")
    print("=" * 50)
    print("\n📝 Test Accounts:")
    print("   - Email: test@example.com, Password: 123456")
    print("   - Email: Umarmirza369@gmail.com, Password: umar13")
    print("=" * 50)
    
    # Run the app with Railway settings
    app.run(debug=DEBUG, port=PORT, host='0.0.0.0')