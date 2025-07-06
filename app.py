from flask import Flask, render_template, request, jsonify, send_file
import yfinance as yf
import plotly.graph_objs as go
import plotly.utils
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_analyser import PortfolioAnalyzer
from portfolio_visualiser_helper import get_sector_plot
import os

template_dir = 'Templates'
app = Flask(__name__, template_folder=template_dir)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_portfolio_data')
def get_portfolio_data():
    """API endpoint to get portfolio tickers and sectors"""
    return jsonify(portfolio_info)

@app.route('/download_orders')
def download_orders():
    """Download the Orders.csv file"""
    try:
        orders_file = 'Orders.csv'
        if os.path.exists(orders_file):
            return send_file(orders_file, as_attachment=True, download_name='Orders.csv')
        else:
            return jsonify({'error': 'Orders.csv file not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/get_orders')
def get_orders():
    """Get all orders for display"""
    try:
        orders_file = 'Orders.csv'
        if os.path.exists(orders_file):
            df = pd.read_csv(orders_file)
            # Convert to list of dictionaries for JSON response
            orders = df.to_dict('records')
            return jsonify({'orders': orders})
        else:
            return jsonify({'orders': []})
    except Exception as e:
        return jsonify({'error': f'Error reading orders: {str(e)}'}), 500

@app.route('/add_order', methods=['POST'])
def add_order():
    """Add a new order to the Orders.csv file"""
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
        
        orders_file = 'Orders.csv'
        
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
def get_last_order():
    """Get the last order for editing"""
    try:
        orders_file = 'Orders.csv'
        if os.path.exists(orders_file):
            df = pd.read_csv(orders_file)
            if not df.empty:
                last_order = df.iloc[-1].to_dict()
                order_json = jsonify({'order': last_order})
                return order_json
            else:
                return jsonify({'error': 'No orders found'}), 404
        else:
            return jsonify({'error': 'Orders.csv file not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error getting last order: {str(e)}'}), 500

@app.route('/update_last_order', methods=['POST'])
def update_last_order():
    """Update the last order"""
    try:
        orders_file = 'Orders.csv'
        if not os.path.exists(orders_file):
            return jsonify({'error': 'Orders.csv file not found'}), 404
        
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

@app.route('/delete_last_order', methods=['POST'])
def delete_last_order():
    """Delete the last order"""
    try:
        orders_file = 'Orders.csv'
        if not os.path.exists(orders_file):
            return jsonify({'error': 'Orders.csv file not found'}), 404
        
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
                    portfolio_plot = get_sector_plot(portfolio_data)
                
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
                    sector_plot = get_sector_plot(sector_data)
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