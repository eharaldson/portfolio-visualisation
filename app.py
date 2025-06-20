from flask import Flask, render_template, request, jsonify
import yfinance as yf
import plotly.graph_objs as go
import plotly.utils
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_analyser import PortfolioAnalyzer

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
                        trace_names.append(f"{sector} (Sector Avg)")
        
        # Add portfolio if requested
        portfolio_data = None
        if plot_portfolio:
            try:
                analyzer = PortfolioAnalyzer('OrdersFiltered.csv')
                portfolio_df = analyzer.calculate_portfolio_performance(start_date, end_date)
                if portfolio_df is not None and not portfolio_df.empty:
                    portfolio_data = portfolio_df
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
                        if not data.empty:
                            sector_data[t] = data['Close']
                    except Exception as e:
                        print(f"Error fetching {t}: {e}")
                
                if sector_data:
                    # Calculate sector average (normalized)
                    normalized_data = {}
                    for t, prices in sector_data.items():
                        if len(prices) > 0:
                            normalized = (prices / prices.iloc[0]) * 100
                            normalized_data[t] = normalized
                    
                    # Calculate average
                    df_normalized = pd.DataFrame(normalized_data)
                    sector_avg = df_normalized.mean(axis=1)
                    all_data[('sector', sector_name)] = sector_avg
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
        if portfolio_data is not None:
            fig.add_trace(go.Scatter(
                x=portfolio_data['date'],
                y=portfolio_data['return_normalized'] * 100,  # Convert to percentage
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
                name = f"{key[1]} (Sector Avg)"
                line_style = dict(width=2, dash='dash', color=colors[color_index % len(colors)])
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