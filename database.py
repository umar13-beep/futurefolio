# backend/database.py - Simple File-based Database
import json
import os
from datetime import datetime

DB_FOLDER = 'database'
USERS_FILE = os.path.join(DB_FOLDER, 'users.json')
PORTFOLIO_FILE = os.path.join(DB_FOLDER, 'portfolio.json')

# Create database folder if not exists
os.makedirs(DB_FOLDER, exist_ok=True)

def init_database():
    """Initialize database files"""
    for file in [USERS_FILE, PORTFOLIO_FILE]:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                json.dump({}, f)

def load_data(filename):
    """Load JSON data from file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_data(filename, data):
    """Save data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

# User Management
def create_user(email, password, name):
    """Create new user"""
    users = load_data(USERS_FILE)
    
    if email in users:
        return False, "User already exists"
    
    users[email] = {
        'email': email,
        'password': password,  # In real app, hash this!
        'name': name,
        'created_at': datetime.now().isoformat(),
        'balance': 100000  # Starting balance
    }
    
    save_data(USERS_FILE, users)
    return True, "User created successfully"

def authenticate_user(email, password):
    """Authenticate user login"""
    users = load_data(USERS_FILE)
    
    if email not in users:
        return False, "User not found"
    
    if users[email]['password'] != password:
        return False, "Invalid password"
    
    return True, "Login successful"

def get_user(email):
    """Get user details"""
    users = load_data(USERS_FILE)
    return users.get(email)

# Portfolio Management
def add_to_portfolio(email, symbol, name, quantity, price):
    """Add stock to user's portfolio"""
    portfolio = load_data(PORTFOLIO_FILE)
    
    if email not in portfolio:
        portfolio[email] = []
    
    transaction = {
        'symbol': symbol,
        'name': name,
        'quantity': quantity,
        'buy_price': price,
        'total_invested': quantity * price,
        'date': datetime.now().isoformat(),
        'transaction_id': len(portfolio[email]) + 1
    }
    
    portfolio[email].append(transaction)
    save_data(PORTFOLIO_FILE, portfolio)
    return True, "Stock added to portfolio"

def get_portfolio(email):
    """Get user's portfolio"""
    portfolio = load_data(PORTFOLIO_FILE)
    return portfolio.get(email, [])

def calculate_portfolio_value(email):
    """Calculate total portfolio value"""
    portfolio = get_portfolio(email)
    total_value = 0
    total_invested = 0
    
    for item in portfolio:
        total_invested += item['total_invested']
        # For demo, assume current price is 5% more than buy price
        current_value = item['total_invested'] * 1.05
        total_value += current_value
    
    return {
        'total_invested': total_invested,
        'current_value': total_value,
        'profit_loss': total_value - total_invested,
        'profit_percentage': ((total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0
    }

# Initialize database on import
init_database()