import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# From an input dictionary of ticker price data, generate a plot of sector performance by checking 
# order file to get the correct weightings for the sector.
def get_sector_plot(sector_data: dict[str, pd.DataFrame]) -> list[float]:
    """
    Generate a list of weighted changes for each ticker in the sector based on order data.
    Args:
        sector_data (dict[str, pd.DataFrame]): Dictionary where keys are sector names and values are DataFrames
            containing ticker price data with 'close_change' column.
    Returns:
        list[float]: List of weighted changes for each ticker in the sector.
    """
    # Read the order data from CSV
    df_orders = pd.read_csv('OrdersFiltered.csv')
    df_orders['date'] = pd.to_datetime(df_orders['date'], format='%d/%m/%Y')

    sector_tickers = list(sector_data.keys())

    sector_normalised = [100]

    # Ensure we are looking at the longest length of data to catch all historical data
    max_length_index = None
    max_length = 0

    for ticker in sector_tickers:
        length_data = len(sector_data[ticker])
        if length_data > max_length:
            max_length = length_data
            max_length_index = sector_data[ticker].index

    # Loop through and add weighting for each ticker in the sector
    for timestamp in max_length_index:

        # Get order data before the current date
        t = timestamp.date()
        current_datetime = datetime(t.year, t.month, t.day)
        current_order_data = df_orders[df_orders['date'] <= current_datetime]

        # Calculate the value of the shares owned for each ticker
        ticker_value = {ticker: current_order_data[current_order_data['ticker'] == ticker]['num_shares'].sum() for ticker in sector_tickers}

        # Calculate weighting of share ownership as fraction of total sector value
        ticker_weightings = {ticker: ticker_value[ticker] / sum(ticker_value.values()) if sum(ticker_value.values()) > 0 else 0 for ticker in sector_tickers}

        # Calculate weighted change for each ticker
        weighted_change = 0
        for ticker in sector_tickers:
            if timestamp in sector_data[ticker].index:
                # Get the close change for the ticker at the current timestamp
                close_change = sector_data[ticker].loc[timestamp]['Close_change']
                weighted_change += close_change * ticker_weightings[ticker]

        sector_normalised.append(sector_normalised[-1] * (1 + weighted_change))

    # Get the data in a series with date in the index for plotting
    sector_normalised = pd.Series(sector_normalised[1:], index=sector_data[sector_tickers[0]].index)

    return sector_normalised

if __name__ == "__main__":

    df_cases = pd.read_csv('CaseStocks.csv')
    sector_tickers = df_cases['Ticker'].tolist()
    # sector_tickers = ['CNQ']

    print(f"Sector tickers: {sector_tickers}")

    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    print(f"Start date: {start_date}")
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Example usage
    sector_data = {}
    for t in sector_tickers:
        print(f"Fetching data for {t}...")
        # try:
        stock = yf.Ticker(t)
        data = stock.history(start=start_date, end=end_date)
        data['Close_change'] = data['Close'].pct_change().fillna(0)

        if not data.empty:
            sector_data[t] = data[['Close', 'Close_change']]
        # except Exception as e:
        #     print(f"Error fetching {t}: {e}")
    

    if sector_data:
        # Calculate sector plot by weighting of tickers
        sector_plot = get_sector_plot(sector_data)
    
    plt.plot(sector_plot)
    plt.show()