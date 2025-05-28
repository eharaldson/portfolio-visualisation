from flask import Flask, render_template, request, jsonify
import yfinance as yf
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime, timedelta

template_dir = 'Templates'
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    # Get form data
    ticker = request.form.get('ticker', '').strip().upper() or 'AAPL'
    start_date = request.form.get('start_date', '').strip()
    end_date = request.form.get('end_date', '').strip()
    
    # Set default dates
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return jsonify({'error': f'No data found for ticker {ticker}'}), 400
        
        # Create plotly figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=ticker,
            line=dict(width=2, color='#1f77b4'),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Close: $%{y:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{ticker} Stock Price',
            xaxis_title='Date',
            yaxis_title='Close Price ($)',
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
        
        # Style the plot with rounded corners effect
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
        return jsonify({'error': f'Error fetching data: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)