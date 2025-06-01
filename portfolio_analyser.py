import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TODO: Check on the erroneous data for these tickers
tickers_to_ignore = ['IEMG']

class PortfolioAnalyzer:
    def __init__(self, transactions_file):
        """
        Initialize with transactions CSV file
        Expected columns: date, ticker, order_type, shares, order_price
        """
        self.transactions = pd.read_csv(transactions_file)
        self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        self.transactions = self.transactions.sort_values('date')
        self.transactions = self.transactions[self.transactions['ticker'].apply(lambda x: x not in tickers_to_ignore)]
        
    def calculate_holdings_over_time(self, start_date, end_date=None):
        """
        Calculate daily holdings and portfolio composition
        """
        if end_date is None:
            end_date = datetime.now().date()
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter transactions within date range
        relevant_transactions = self.transactions[
            self.transactions['date'] <= end_date
        ].copy()
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Get unique tickers
        tickers = relevant_transactions['ticker'].unique()
        
        # Calculate holdings for each day
        holdings_data = {}
        
        # Loop through each date in range and calulate holdings changes
        for date in date_range:
            
            daily_holdings = {}
            
            # Get transactions up to this date
            transactions_to_date = relevant_transactions[
                relevant_transactions['date'] <= date
            ]
            
            for ticker in tickers:
                ticker_transactions = transactions_to_date[
                    transactions_to_date['ticker'] == ticker
                ]
                # Calculate net shares
                buys = ticker_transactions[
                    (ticker_transactions['ticker'] == ticker) & (ticker_transactions['order_type'] == 'buy')
                ]['num_shares'].sum()
                sells = ticker_transactions[
                    (ticker_transactions['ticker'] == ticker) & (ticker_transactions['order_type'] == 'sell')
                ]['num_shares'].sum()
                
                net_shares = buys - sells
                
                if buys - sells < 0:
                    print(f"Warning: Negative shares for {ticker} on {start_date}. Check your transactions.")

                if net_shares > 0:
                    # Calculate average cost basis
                    buy_transactions = ticker_transactions[
                        ticker_transactions['order_type'] == 'buy'
                    ]
                    if not buy_transactions.empty:
                        total_cost = (buy_transactions['num_shares'] * 
                                    buy_transactions['order_price']).sum()
                        total_shares_bought = buy_transactions['num_shares'].sum()
                        avg_cost = total_cost / total_shares_bought if total_shares_bought > 0 else 0
                    else:
                        avg_cost = 0
                    
                    daily_holdings[ticker] = {
                        'num_shares': net_shares,
                        'avg_cost': avg_cost
                    }
            
            holdings_data[date] = daily_holdings
        
        return holdings_data, tickers
    
    def get_historical_prices(self, tickers, start_date, end_date=None) -> list[pd.DataFrame]:
        """
        Fetch historical prices for all tickers
        """
        if end_date is None:
            end_date = datetime.now()
        
        price_data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)

                if not hist.empty:
                    price_data[ticker] = hist['Close']
                else:
                    print(f"No data found for {ticker}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        return price_data
    
    def calculate_portfolio_performance(self, start_date, end_date=None):
        """
        Calculate daily portfolio value, cash flows, and returns
        """
        holdings_data, all_tickers = self.calculate_holdings_over_time(start_date, end_date)
        
        if not holdings_data:
            return None
        
        # Fetch historical prices
        price_data = self.get_historical_prices(all_tickers, start_date, end_date)
        
        # Calculate daily portfolio values
        portfolio_data = []
        cumulative_invested = 0

        dates = price_data[all_tickers[0]].index

        return_normalized = 1

        for date in dates:

            date_no_timezone = date.replace(tzinfo=None)
            days_holdings = holdings_data[date_no_timezone]
            
            daily_value = 0
            daily_cost_basis = 0
            
            # Calculate cash flows for this date
            daily_transactions = self.transactions[
                self.transactions['date'] == date
            ]
            daily_cash_flow = 0
            
            for _, transaction in daily_transactions.iterrows():
                if transaction['order_type'] == 'buy':
                    daily_cash_flow -= transaction['num_shares'] * transaction['order_price']
                elif transaction['order_type'] == 'sell':
                    daily_cash_flow += transaction['num_shares'] * transaction['order_price']
            
            cumulative_invested += daily_cash_flow
            
            # Calculate portfolio value
            for ticker, holding_info in days_holdings.items():
                shares = holding_info['num_shares']
                avg_cost = holding_info['avg_cost']
                
                # Get current price
                if ticker in price_data and date in price_data[ticker].index:
                    current_price = price_data[ticker][date]
                    daily_value += shares * current_price
                    daily_cost_basis += shares * avg_cost
                elif ticker in price_data and len(price_data[ticker]) > 0:
                    # Use the last available price
                    available_prices = price_data[ticker][price_data[ticker].index <= date]
                    if len(available_prices) > 0:
                        current_price = available_prices.iloc[-1]
                        daily_value += shares * current_price
                        daily_cost_basis += shares * avg_cost
            
            if len(portfolio_data) == 0:
                return_normalized = 1
            else:
                return_normalized *= 1 + ((daily_value - portfolio_data[-1]['portfolio_value']) / portfolio_data[-1]['portfolio_value'])

            portfolio_data.append({
                'date': date,
                'portfolio_value': daily_value,
                'cost_basis': daily_cost_basis,
                'cumulative_invested': abs(cumulative_invested),
                'unrealized_pnl': daily_value - daily_cost_basis,
                'return_normalized': return_normalized
            })
        
        return pd.DataFrame(portfolio_data)
    
    def plot_portfolio_performance(self, start_date, end_date=None, benchmark_ticker='SPY'):
        """
        Create comprehensive portfolio performance plots
        """
        df = self.calculate_portfolio_performance(start_date, end_date)
        
        if df is None or df.empty:
            print("No data available for the specified date range")
            return
            
        # Get benchmark data
        benchmark_data = None
        if benchmark_ticker:
            try:
                benchmark = yf.Ticker(benchmark_ticker)
                benchmark_hist = benchmark.history(start=start_date, end=end_date)
                if not benchmark_hist.empty:
                    # Calculate benchmark returns
                    benchmark_start_price = benchmark_hist['Close'].iloc[0]
                    benchmark_data = ((benchmark_hist['Close'] / benchmark_start_price) - 1) * 100
            except Exception as e:
                print(f"Error fetching benchmark data: {e}")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Portfolio Value Over Time', 'Total Return %', 'Unrealized P&L'],
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Portfolio value plot
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#2E86AB', width=3),
                hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['cumulative_invested'],
                mode='lines',
                name='Amount Invested',
                line=dict(color='#A23B72', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Invested: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Total return percentage
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['return_normalized'],
                mode='lines',
                name='Portfolio Return %',
                line=dict(color='#F18F01', width=3),
                hovertemplate='Date: %{x}<br>Return: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add benchmark if available
        if benchmark_data is not None:
            # Align benchmark data with portfolio dates
            aligned_benchmark = []
            for date in df['date']:
                if date in benchmark_data.index:
                    aligned_benchmark.append(benchmark_data[date])
                else:
                    # Find closest earlier date
                    earlier_dates = benchmark_data.index[benchmark_data.index <= date]
                    if len(earlier_dates) > 0:
                        aligned_benchmark.append(benchmark_data[earlier_dates[-1]])
                    else:
                        aligned_benchmark.append(0)
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=aligned_benchmark,
                    mode='lines',
                    name=f'{benchmark_ticker} Return %',
                    line=dict(color='#C73E1D', width=2, dash='dot'),
                    hovertemplate=f'Date: %{{x}}<br>{benchmark_ticker}: %{{y:.1f}}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Unrealized P&L
        colors = ['green' if x >= 0 else 'red' for x in df['unrealized_pnl']]
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['unrealized_pnl'],
                name='Unrealized P&L',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='Date: %{x}<br>P&L: $%{y:,.0f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Portfolio Performance Analysis (Starting {start_date})',
            height=800,
            showlegend=True,
            plot_bgcolor='#f5f5f5',
            paper_bgcolor='white',
            font=dict(color='#333333'),
            hovermode='x unified'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        # Style the subplots
        for i in range(1, 4):
            fig.update_xaxes(showgrid=True, gridcolor='white', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='white', row=i, col=1)
        
        fig.show()
        
        # Print summary statistics
        if not df.empty:
            final_value = df['portfolio_value'].iloc[-1]
            total_invested = df['cumulative_invested'].iloc[-1]
            total_return = df['total_return_pct'].iloc[-1]
            max_value = df['portfolio_value'].max()
            min_value = df['portfolio_value'].min()
            
            print("\n" + "="*50)
            print("PORTFOLIO PERFORMANCE SUMMARY")
            print("="*50)
            print(f"Period: {df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
            print(f"Total Invested: ${total_invested:,.0f}")
            print(f"Current Value: ${final_value:,.0f}")
            print(f"Total Return: {total_return:.1f}%")
            print(f"Absolute Gain/Loss: ${final_value - total_invested:,.0f}")
            print(f"Highest Portfolio Value: ${max_value:,.0f}")
            print(f"Lowest Portfolio Value: ${min_value:,.0f}")
            
            if benchmark_data is not None and len(aligned_benchmark) > 0:
                benchmark_return = aligned_benchmark[-1]
                print(f"{benchmark_ticker} Return: {benchmark_return:.1f}%")
                print(f"Outperformance vs {benchmark_ticker}: {total_return - benchmark_return:.1f}%")

if __name__ == "__main__":

    # Initialize the PortfolioAnalyzer with the transactions CSV file
    analyzer = PortfolioAnalyzer('OrdersFiltered.csv')
    
    # Plot performance from a specific start date
    analyzer.plot_portfolio_performance(
        start_date='2024-01-01',
        end_date=None,  # Use current date
        benchmark_ticker=None#'SPY'  # Compare against S&P 500
    )
