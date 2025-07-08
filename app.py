from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import yfinance as yf
import plotly.graph_objs as go
import plotly.utils
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_analyser import PortfolioAnalyzer
from portfolio_visualiser_helper import get_sector_plot, get_sector_portfolio_plot, get_stock_portfolio_allocation
import os
import csv
import tempfile

template_dir = 'templates'
app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production!
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the dashboard.'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, email, password_hash):
        self.id = str(user_id)  # Ensure ID is always string
        self.email = email
        self.password_hash = password_hash
    
    def is_authenticated(self):
        return True
    
    def is_active(self):
        return True
    
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)

# User management functions
def init_user_files():
    """Initialize user management files if they don't exist"""
    users_file = 'users.csv'
    user_orders_file = 'user_orders_mapping.csv'
    
    if not os.path.exists(users_file):
        with open(users_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'email', 'password_hash', 'created_at'])
    
    if not os.path.exists(user_orders_file):
        with open(user_orders_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'orders_file'])

def get_user_by_email(email):
    """Get user by email from CSV"""
    users_file = 'users.csv'
    if not os.path.exists(users_file):
        return None
    
    df = pd.read_csv(users_file)
    user_row = df[df['email'] == email]
    
    if not user_row.empty:
        row = user_row.iloc[0]
        return User(row['user_id'], row['email'], row['password_hash'])
    return None

def get_user_by_id(user_id):
    """Get user by ID from CSV"""
    users_file = 'users.csv'
    if not os.path.exists(users_file):
        return None
    
    try:
        df = pd.read_csv(users_file)
        # Convert user_id to string for comparison since CSV stores as string
        user_row = df[df['user_id'].astype(str) == str(user_id)]
        
        if not user_row.empty:
            row = user_row.iloc[0]
            return User(str(row['user_id']), row['email'], row['password_hash'])
        return None
    except Exception as e:
        print(f"Error in get_user_by_id: {e}")
        return None

def create_user(email, password):
    """Create a new user"""
    users_file = 'users.csv'
    user_orders_file = 'user_orders_mapping.csv'
    
    # Check if user already exists
    if get_user_by_email(email):
        return None
    
    # Generate user ID (timestamp-based)
    user_id = str(int(datetime.now().timestamp()))
    password_hash = generate_password_hash(password)
    created_at = datetime.now().isoformat()
    
    # Add user to users.csv
    with open(users_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, email, password_hash, created_at])
    
    # Create user-specific orders file and add mapping
    orders_filename = f'Orders_{user_id}.csv'
    with open(user_orders_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, orders_filename])
    
    # Create empty orders file for the user
    with open(orders_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'num_shares', 'company', 'ticker', 'price', 'foreign_commission', 'domestic_commission', 'order_type', 'split_ratio', 'merger_old_ticker'])
    
    return User(str(user_id), email, password_hash)

def get_user_orders_file(user_id):
    """Get the orders file for a specific user"""
    user_orders_file = 'user_orders_mapping.csv'
    if not os.path.exists(user_orders_file):
        return None
    
    try:
        df = pd.read_csv(user_orders_file)
        # Convert user_id to string for comparison
        mapping_row = df[df['user_id'].astype(str) == str(user_id)]
        
        if not mapping_row.empty:
            return mapping_row.iloc[0]['orders_file']
        return None
    except Exception as e:
        print(f"Error in get_user_orders_file: {e}")
        return None


from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import yfinance as yf
import plotly.graph_objs as go
import plotly.utils
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_analyser import PortfolioAnalyzer
from portfolio_visualiser_helper import get_sector_plot, get_stock_portfolio_allocation
import os
import csv
import tempfile

template_dir = 'Templates'
app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production!
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the dashboard.'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, email, password_hash):
        self.id = str(user_id)  # Ensure ID is always string
        self.email = email
        self.password_hash = password_hash
    
    def is_authenticated(self):
        return True
    
    def is_active(self):
        return True
    
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)

# User management functions
def init_user_files():
    """Initialize user management files if they don't exist"""
    users_file = 'users.csv'
    user_orders_file = 'user_orders_mapping.csv'
    
    if not os.path.exists(users_file):
        with open(users_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'email', 'password_hash', 'created_at'])
    
    if not os.path.exists(user_orders_file):
        with open(user_orders_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'orders_file'])

def get_user_by_email(email):
    """Get user by email from CSV"""
    users_file = 'users.csv'
    if not os.path.exists(users_file):
        return None
    
    df = pd.read_csv(users_file)
    user_row = df[df['email'] == email]
    
    if not user_row.empty:
        row = user_row.iloc[0]
        return User(row['user_id'], row['email'], row['password_hash'])
    return None

def get_user_by_id(user_id):
    """Get user by ID from CSV"""
    users_file = 'users.csv'
    if not os.path.exists(users_file):
        return None
    
    try:
        df = pd.read_csv(users_file)
        # Convert user_id to string for comparison since CSV stores as string
        user_row = df[df['user_id'].astype(str) == str(user_id)]
        
        if not user_row.empty:
            row = user_row.iloc[0]
            return User(str(row['user_id']), row['email'], row['password_hash'])
        return None
    except Exception as e:
        print(f"Error in get_user_by_id: {e}")
        return None

def create_user(email, password):
    """Create a new user"""
    users_file = 'users.csv'
    user_orders_file = 'user_orders_mapping.csv'
    
    # Check if user already exists
    if get_user_by_email(email):
        return None
    
    # Generate user ID (timestamp-based)
    user_id = str(int(datetime.now().timestamp()))
    password_hash = generate_password_hash(password)
    created_at = datetime.now().isoformat()
    
    # Add user to users.csv
    with open(users_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, email, password_hash, created_at])
    
    # Create user-specific orders file and add mapping
    orders_filename = f'Orders_{user_id}.csv'
    with open(user_orders_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, orders_filename])
    
    # Create empty orders file for the user
    with open(orders_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'num_shares', 'company', 'ticker', 'price', 'foreign_commission', 'domestic_commission', 'order_type', 'split_ratio', 'merger_old_ticker'])
    
    return User(str(user_id), email, password_hash)

def get_user_orders_file(user_id):
    """Get the orders file for a specific user"""
    user_orders_file = 'user_orders_mapping.csv'
    if not os.path.exists(user_orders_file):
        return None
    
    try:
        df = pd.read_csv(user_orders_file)
        # Convert user_id to string for comparison
        mapping_row = df[df['user_id'].astype(str) == str(user_id)]
        
        if not mapping_row.empty:
            return mapping_row.iloc[0]['orders_file']
        return None
    except Exception as e:
        print(f"Error in get_user_orders_file: {e}")
        return None

@login_manager.user_loader
def load_user(user_id):
    print(f"Loading user with ID: {user_id} (type: {type(user_id)})")  # Debug
    # Flask-Login passes user_id as string, make sure we handle it correctly
    user = get_user_by_id(str(user_id))
    print(f"Loaded user: {user.email if user else 'None'}")  # Debug
    return user

def cleanup_temp_files():
    """Clean up temporary upload files older than 1 hour"""
    try:
        temp_dir = app.config['UPLOAD_FOLDER']
        current_time = datetime.now()
        
        for filename in os.listdir(temp_dir):
            if filename.startswith('temp_upload_'):
                filepath = os.path.join(temp_dir, filename)
                if os.path.isfile(filepath):
                    # Check if file is older than 1 hour
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    if (current_time - file_time).total_seconds() > 3600:  # 1 hour
                        try:
                            os.remove(filepath)
                            print(f"Cleaned up old temp file: {filename}")
                        except:
                            pass
    except Exception as e:
        print(f"Error during temp file cleanup: {e}")

# Initialize user files on startup
init_user_files()
# Clean up any old temp files
cleanup_temp_files()

# Load portfolio data on startup
def load_portfolio_info():
    """Load ticker and sector information from CSV"""
    try:
        df = pd.read_csv('CaseStocksSectors.csv')
        tickers = df['ticker'].tolist()
        
        # Get all sector columns (all columns except 'ticker')
        sector_columns = [col for col in df.columns if col != 'ticker']
        
        # Create sector mapping
        sector_mapping = {}
        for sector in sector_columns:
            sector_mapping[sector] = df[df[sector] == True]['ticker'].tolist()
        
        return {
            'tickers': tickers,
            'sectors': sector_columns,
            'sector_mapping': sector_mapping
        }
    except Exception as e:
        print(f"Error loading portfolio info: {e}")
        return {
            'tickers': [],
            'sectors': [],
            'sector_mapping': {}
        }

portfolio_info = load_portfolio_info()

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = get_user_by_email(email)
        
        if user and check_password_hash(user.password_hash, password):
            print(f"Password verified for user: {email}")  # Debug
            login_user(user, remember=True)  # Add remember=True
            print(f"login_user() called")  # Debug
            print(f"current_user.is_authenticated after login: {current_user.is_authenticated}")  # Debug
            print(f"current_user.id after login: {current_user.id}")  # Debug
            redirect_url = url_for('index')
            print(f"Generated redirect URL: {redirect_url}")  # Debug
            response = redirect(redirect_url)
            print(f"Redirect response: {response}")  # Debug
            print(f"Response headers: {response.headers}")  # Debug
            return response
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long')
            return render_template('register.html')
        
        user = create_user(email, password)
        
        if user:
            login_user(user)
            flash('Registration successful!')
            return redirect(url_for('index'))
        else:
            flash('Email already exists')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# routes (require login)
@app.route('/')
@login_required
def index():
    print(f"Index route called for user: {current_user.email if current_user.is_authenticated else 'Not authenticated'}")  # Debug
    print(f"Current user authenticated: {current_user.is_authenticated}")  # Debug
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}")  # Debug
        return f"Error: {e}", 500

@app.route('/get_portfolio_data')
@login_required
def get_portfolio_data():
    """API endpoint to get portfolio tickers and sectors"""
    return jsonify(portfolio_info)

@app.route('/get_portfolio_allocation')
@login_required
def get_portfolio_allocation():
    """Get portfolio allocation data for pie chart"""
    try:
        orders_file = get_user_orders_file(current_user.id)
        allocation_data = get_stock_portfolio_allocation(orders_file)
        return jsonify({'allocation': allocation_data})
    except Exception as e:
        return jsonify({'error': f'Error getting portfolio allocation: {str(e)}'}), 500

@app.route('/download_orders')
@login_required
def download_orders():
    """Download the user's Orders.csv file"""
    try:
        orders_file = get_user_orders_file(current_user.id)
        if orders_file and os.path.exists(orders_file):
            return send_file(orders_file, as_attachment=True, download_name='My_Orders.csv')
        else:
            return jsonify({'error': 'Orders file not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/get_orders')
@login_required
def get_orders():
    """Get all orders for the current user"""
    try:
        orders_file = get_user_orders_file(current_user.id)
        if orders_file and os.path.exists(orders_file):
            df = pd.read_csv(orders_file)
            # Convert to list of dictionaries for JSON response
            orders = df.to_dict('records')
            return jsonify({'orders': orders})
        else:
            return jsonify({'orders': []})
    except Exception as e:
        return jsonify({'error': f'Error reading orders: {str(e)}'}), 500

@app.route('/add_order', methods=['POST'])
@login_required
def add_order():
    """Add a new order to the user's Orders.csv file"""
    try:
        # Get form data
        order_data = {
            'date': request.form.get('date'),
            'num_shares': float(request.form.get('num_shares')),
            'company': request.form.get('company'),
            'ticker': request.form.get('ticker').upper(),
            'price': float(request.form.get('price')),
            'foreign_commission': float(request.form.get('foreign_commission', 0)),
            'domestic_commission': float(request.form.get('domestic_commission', 0)),
            'order_type': request.form.get('order_type'),
            'split_ratio': request.form.get('split_ratio', ''),
            'merger_old_ticker': request.form.get('merger_old_ticker', '')
        }
        
        orders_file = get_user_orders_file(current_user.id)
        
        if not orders_file:
            return jsonify({'error': 'Orders file not found for user'}), 404
        
        # Check if file exists
        if os.path.exists(orders_file):
            # Read existing data
            df = pd.read_csv(orders_file)
            # Append new order
            new_order_df = pd.DataFrame([order_data])
            df = pd.concat([df, new_order_df], ignore_index=True)
        else:
            # Create new file
            df = pd.DataFrame([order_data])
        
        # Save to CSV
        df.to_csv(orders_file, index=False)
        
        return jsonify({'success': True, 'message': 'Order added successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Error adding order: {str(e)}'}), 500

@app.route('/get_last_order')
@login_required
def get_last_order():
    """Get the last order for editing"""
    try:
        orders_file = get_user_orders_file(current_user.id)
        if orders_file and os.path.exists(orders_file):
            df = pd.read_csv(orders_file)
            if not df.empty:
                last_order = df.iloc[-1].to_dict()
                return jsonify({'order': last_order})
            else:
                return jsonify({'error': 'No orders found'}), 404
        else:
            return jsonify({'error': 'Orders file not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error getting last order: {str(e)}'}), 500

@app.route('/update_last_order', methods=['POST'])
@login_required
def update_last_order():
    """Update the last order"""
    try:
        orders_file = get_user_orders_file(current_user.id)
        if not orders_file or not os.path.exists(orders_file):
            return jsonify({'error': 'Orders file not found'}), 404
        
        df = pd.read_csv(orders_file)
        if df.empty:
            return jsonify({'error': 'No orders to update'}), 404
        
        # Update the last row with new data
        order_data = {
            'date': request.form.get('date'),
            'num_shares': float(request.form.get('num_shares')),
            'company': request.form.get('company'),
            'ticker': request.form.get('ticker').upper(),
            'price': float(request.form.get('price')),
            'foreign_commission': float(request.form.get('foreign_commission', 0)),
            'domestic_commission': float(request.form.get('domestic_commission', 0)),
            'order_type': request.form.get('order_type'),
            'split_ratio': request.form.get('split_ratio', ''),
            'merger_old_ticker': request.form.get('merger_old_ticker', '')
        }
        
        # Update last row
        for key, value in order_data.items():
            df.iloc[-1, df.columns.get_loc(key)] = value
        
        # Save to CSV
        df.to_csv(orders_file, index=False)
        
        return jsonify({'success': True, 'message': 'Order updated successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Error updating order: {str(e)}'}), 500

def validate_csv_format(df):
    """Validate that the CSV has the correct columns and format"""
    required_columns = [
        'date', 'num_shares', 'company', 'ticker', 'price', 
        'foreign_commission', 'domestic_commission', 'order_type', 
        'split_ratio', 'merger_old_ticker'
    ]
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if there are extra columns
    extra_columns = [col for col in df.columns if col not in required_columns]
    if extra_columns:
        return False, f"Unexpected columns found: {', '.join(extra_columns)}"
    
    # Check if DataFrame is empty
    if df.empty:
        return False, "CSV file is empty"
    
    # Validate data types and required fields
    try:
        # Check for required fields (non-empty)
        for index, row in df.iterrows():
            if pd.isna(row['date']) or str(row['date']).strip() == '':
                return False, f"Row {index + 1}: Date is required"
            if pd.isna(row['ticker']) or str(row['ticker']).strip() == '':
                return False, f"Row {index + 1}: Ticker is required"
            if pd.isna(row['company']) or str(row['company']).strip() == '':
                return False, f"Row {index + 1}: Company is required"
            if pd.isna(row['order_type']) or str(row['order_type']).strip() == '':
                return False, f"Row {index + 1}: Order type is required"
            
            # Validate order_type values
            if str(row['order_type']).lower() not in ['buy', 'sell', 'split', 'merger']:
                return False, f"Row {index + 1}: Order type must be 'buy' or 'sell' or 'split' or 'merger'"
            
            # Validate numeric fields (num_shares and price are required)
            try:
                if pd.isna(row['num_shares']) or str(row['num_shares']).strip() == '':
                    return False, f"Row {index + 1}: Number of shares is required"
                if pd.isna(row['price']) or str(row['price']).strip() == '':
                    return False, f"Row {index + 1}: Price is required"
                
                float(row['num_shares'])
                float(row['price'])
                
                # Commission fields can be empty/NaN, but if present should be numeric
                if not pd.isna(row['foreign_commission']) and str(row['foreign_commission']).strip() != '':
                    float(row['foreign_commission'])
                if not pd.isna(row['domestic_commission']) and str(row['domestic_commission']).strip() != '':
                    float(row['domestic_commission'])
            except (ValueError, TypeError):
                return False, f"Row {index + 1}: Invalid numeric values in num_shares, price, or commission fields"
            
            # Validate date format (try to parse it)
            try:
                pd.to_datetime(row['date'])
            except:
                return False, f"Row {index + 1}: Invalid date format. Use YYYY-MM-DD or similar standard format"
    
    except Exception as e:
        return False, f"Error validating data: {str(e)}"
    
    return True, "CSV format is valid"

@app.route('/upload_orders', methods=['POST'])
@login_required
def upload_orders():
    """Upload and validate CSV file of orders"""
    print("upload_orders endpoint called")
    try:
        print("Checking for file in request...")
        if 'file' not in request.files:
            print("No file in request.files")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"File object: {file}")
        print(f"Filename: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            print("File is not CSV")
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Read and validate the CSV
        try:
            print("Reading CSV file...")
            df = pd.read_csv(file)
            print(f"CSV read successfully. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Validate CSV format
        print("Validating CSV format...")
        is_valid, message = validate_csv_format(df)
        print(f"Validation result: {is_valid}, Message: {message}")
        
        if not is_valid:
            print("Validation failed")
            return jsonify({'error': message}), 400
        
        # Store the validated data temporarily in a file instead of session
        print("Storing data in temporary file...")
        temp_filename = f"temp_upload_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Save the cleaned data to temp file
        df_clean = df.copy()
        df_clean['foreign_commission'] = df_clean['foreign_commission'].fillna(0)
        df_clean['domestic_commission'] = df_clean['domestic_commission'].fillna(0)
        df_clean['split_ratio'] = df_clean['split_ratio'].fillna('')
        df_clean['merger_old_ticker'] = df_clean['merger_old_ticker'].fillna('')
        df_clean['ticker'] = df_clean['ticker'].str.upper()
        df_clean['order_type'] = df_clean['order_type'].str.lower()
        
        df_clean.to_csv(temp_filepath, index=False)
        
        # Store only the temp filename in session (much smaller)
        session['pending_upload'] = {
            'temp_file': temp_filename,
            'original_filename': secure_filename(file.filename),
            'row_count': len(df)
        }
        print("Temp filename stored in session successfully")
        
        # Create preview data with proper NaN handling
        preview_df = df.head(3).fillna('')  # Replace NaN with empty strings
        preview_data = preview_df.to_dict('records')
        
        response_data = {
            'success': True, 
            'message': 'CSV validated successfully',
            'row_count': len(df),
            'preview': preview_data
        }
        print(f"Returning response: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Exception in upload_orders: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing upload: {str(e)}'}), 500

@app.route('/confirm_upload', methods=['POST'])
@login_required
def confirm_upload():
    """Confirm and apply the uploaded CSV data"""
    try:
        if 'pending_upload' not in session:
            return jsonify({'error': 'No pending upload found'}), 400
        
        # Get the stored data
        upload_data = session['pending_upload']
        temp_filename = upload_data['temp_file']
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Check if temp file exists
        if not os.path.exists(temp_filepath):
            return jsonify({'error': 'Temporary file not found. Please upload again.'}), 400
        
        # Read the cleaned data from temp file
        df = pd.read_csv(temp_filepath)
        
        # Get user's orders file
        orders_file = get_user_orders_file(current_user.id)
        if not orders_file:
            return jsonify({'error': 'User orders file not found'}), 404
        
        # Backup existing file if it exists and has data
        backup_created = False
        if os.path.exists(orders_file):
            existing_df = pd.read_csv(orders_file)
            if not existing_df.empty:
                backup_file = f"{orders_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                existing_df.to_csv(backup_file, index=False)
                backup_created = True
        
        # Save the new data
        df.to_csv(orders_file, index=False)
        
        # Clean up: remove temp file and clear session data
        try:
            os.remove(temp_filepath)
        except:
            pass  # If temp file removal fails, it's not critical
        
        session.pop('pending_upload', None)
        
        message = f"Successfully uploaded {len(df)} orders"
        if backup_created:
            message += ". Previous orders backed up."
        
        return jsonify({'success': True, 'message': message})
        
    except Exception as e:
        return jsonify({'error': f'Error confirming upload: {str(e)}'}), 500

@app.route('/cancel_upload', methods=['POST'])
@login_required
def cancel_upload():
    """Cancel the pending upload"""
    try:
        if 'pending_upload' in session:
            upload_data = session['pending_upload']
            if 'temp_file' in upload_data:
                temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_data['temp_file'])
                try:
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
                except:
                    pass  # If cleanup fails, it's not critical
        
        session.pop('pending_upload', None)
        return jsonify({'success': True, 'message': 'Upload cancelled'})
    except Exception as e:
        return jsonify({'error': f'Error cancelling upload: {str(e)}'}), 500

@app.route('/delete_last_order', methods=['POST'])
@login_required
def delete_last_order():
    """Delete the last order"""
    try:
        orders_file = get_user_orders_file(current_user.id)
        if not orders_file or not os.path.exists(orders_file):
            return jsonify({'error': 'Orders file not found'}), 404
        
        df = pd.read_csv(orders_file)
        if df.empty:
            return jsonify({'error': 'No orders to delete'}), 404
        
        # Remove the last row
        df = df.iloc[:-1]
        
        # Save to CSV
        df.to_csv(orders_file, index=False)
        
        return jsonify({'success': True, 'message': 'Order deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Error deleting order: {str(e)}'}), 500

@app.route('/plot', methods=['POST'])
@login_required
def plot():
    # Get form data
    ticker = request.form.get('ticker', '').strip().upper()
    start_date = request.form.get('start_date', '').strip()
    end_date = request.form.get('end_date', '').strip()
    selected_tickers = request.form.get('selected_tickers', '').strip()
    selected_sectors = request.form.get('selected_sectors', '').strip()
    plot_portfolio = request.form.get('plot_portfolio', 'false') == 'true'
    
    # Set default dates
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    try:
        # Collect all tickers to plot
        tickers_to_plot = []
        trace_names = []
        
        # Add searched ticker if provided
        if ticker:
            tickers_to_plot.append(ticker)
            trace_names.append(ticker)
        
        # Add selected individual tickers
        if selected_tickers:
            for t in selected_tickers.split(','):
                t = t.strip()
                if t and t not in tickers_to_plot:
                    tickers_to_plot.append(t)
                    trace_names.append(t)
        
        # Add tickers from selected sectors
        if selected_sectors:
            for sector in selected_sectors.split(','):
                sector = sector.strip()
                if sector in portfolio_info['sector_mapping']:
                    sector_tickers = portfolio_info['sector_mapping'][sector]
                    # Create a combined trace for the sector
                    if sector_tickers:
                        # We'll calculate sector average later
                        tickers_to_plot.append(('sector', sector, sector_tickers))
                        trace_names.append(f"{sector} (Sector)")
        
        # Add portfolio if requested
        portfolio_data = {}
        if plot_portfolio:
            try:
                for t in portfolio_info['tickers']:
                    try:
                        stock = yf.Ticker(t)
                        data = stock.history(start=start_date, end=end_date)
                        data['Close_change'] = data['Close'].pct_change().fillna(0)

                        if not data.empty:
                            portfolio_data[t] = data[['Close', 'Close_change']]
                    except Exception as e:
                        print(f"Error fetching {t}: {e}")
                
                if portfolio_data:
                    # Calculate sector plot by weighting of tickers
                    orders_file = get_user_orders_file(current_user.id)
                    portfolio_plot = get_sector_plot(portfolio_data, orders_file)
                
            except Exception as e:
                print(f"Error calculating portfolio performance: {e}")
        
        # Fetch stock data for all tickers
        all_data = {}
        for item in tickers_to_plot:
            
            if isinstance(item, tuple) and item[0] == 'sector':

                # Handle sector - fetch all tickers in the sector
                _, sector_name, sector_tickers = item
                sector_data = {}
                for t in sector_tickers:
                    try:
                        stock = yf.Ticker(t)
                        data = stock.history(start=start_date, end=end_date)
                        data['Close_change'] = data['Close'].pct_change().fillna(0)

                        if not data.empty:
                            sector_data[t] = data[['Close', 'Close_change']]
                    except Exception as e:
                        print(f"Error fetching {t}: {e}")
                
                if sector_data:
                    # Calculate sector plot by weighting of tickers
                    orders_file = get_user_orders_file(current_user.id)
                    sector_plot = get_sector_plot(sector_data, orders_file)
                    all_data[('sector', sector_name)] = sector_plot
            else:
                # Handle individual ticker
                try:
                    stock = yf.Ticker(item)
                    data = stock.history(start=start_date, end=end_date)
                    if not data.empty:
                        # Normalize to 100 at start
                        normalized = (data['Close'] / data['Close'].iloc[0]) * 100
                        all_data[item] = normalized
                except Exception as e:
                    print(f"Error fetching {item}: {e}")
        
        # Create plotly figure
        fig = go.Figure()
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Add portfolio trace first if available
        color_index = 0
        if plot_portfolio:
            fig.add_trace(go.Scatter(
                x=portfolio_plot.index,
                y=portfolio_plot.values,  # Convert to percentage
                mode='lines',
                name='Portfolio',
                line=dict(width=3, color='#2E86AB'),
                hovertemplate='<b>Portfolio</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
            color_index += 1
        
        # Add traces for all data
        for i, (key, data) in enumerate(all_data.items()):
            if isinstance(key, tuple) and key[0] == 'sector':
                name = f"{key[1]} (Sector)"
                line_style = dict(width=2, color=colors[color_index % len(colors)])
            else:
                name = key
                line_style = dict(width=2, color=colors[color_index % len(colors)])
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data.values,
                mode='lines',
                name=name,
                line=line_style,
                hovertemplate=f'<b>{name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
            color_index += 1
        
        # Update layout
        title_parts = []
        if plot_portfolio:
            title_parts.append("Portfolio")
        if ticker:
            title_parts.append(ticker)
        if selected_tickers:
            title_parts.extend([t.strip() for t in selected_tickers.split(',') if t.strip()])
        if selected_sectors:
            title_parts.extend([f"{s.strip()} Sector" for s in selected_sectors.split(',') if s.strip()])
        
        title = "Normalized Performance: " + ", ".join(title_parts[:3])
        if len(title_parts) > 3:
            title += f" (+{len(title_parts)-3} more)"
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Normalized Value (Base = 100)',
            plot_bgcolor='#f5f5f5',
            paper_bgcolor='#f5f5f5',
            font=dict(color='#333333'),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Style the plot
        fig.update_xaxes(
            showgrid=True,
            gridcolor='white',
            linecolor='#cccccc'
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='white',
            linecolor='#cccccc'
        )
        
        # Convert to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({'plot': graphJSON})
        
    except Exception as e:
        return jsonify({'error': f'Error generating plot: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)