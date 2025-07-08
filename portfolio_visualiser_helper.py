import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def get_current_num_shares(current_order_data: pd.DataFrame, ticker: str) -> float:

    num_shares_bought = current_order_data[(current_order_data['ticker'] == ticker) & (current_order_data['order_type'] != 'sell')]['num_shares'].sum()
    num_shares_sold = current_order_data[(current_order_data['ticker'] == ticker) & (current_order_data['order_type'] == 'sell')]['num_shares'].sum()

    return num_shares_bought - num_shares_sold if num_shares_bought > 0 else 0

# From an input dictionary of ticker price data, generate a plot of sector performance by checking 
# order file to get the correct weightings for the sector.
def get_sector_portfolio_plot(ticker_data: dict[str, pd.DataFrame], order_file: str) -> list[float]:
    """
    Generate a list of weighted changes for each ticker in the sector based on order data.
    Args:
        ticker_data (dict[str, pd.DataFrame]): Dictionary where keys are sector names and values are DataFrames
            containing ticker price data with 'close_change' column.
    Returns:
        list[float]: List of weighted changes for each ticker in the sector.
    """
    # Read the order data from CSV
    df_orders = pd.read_csv(order_file)
    df_orders['date'] = pd.to_datetime(df_orders['date'], format='%d/%m/%Y')

    selected_tickers = list(ticker_data.keys())

    sector_normalised = [100]

    # Ensure we are looking at the longest length of data to catch all historical data
    max_length_index = None
    max_length = 0

    for ticker in selected_tickers:
        length_data = len(ticker_data[ticker])
        if length_data > max_length:
            max_length = length_data
            max_length_index = ticker_data[ticker].index

    # Loop through and add weighting for each ticker in the sector
    for timestamp in max_length_index:

        # Get order data before the current date
        t = timestamp.date()
        current_datetime = datetime(t.year, t.month, t.day)
        current_order_data = df_orders[df_orders['date'] <= current_datetime]

        # Calculate the value of the shares owned for each ticker
        ticker_value = {ticker: get_current_num_shares(current_order_data=current_order_data, ticker=ticker) for ticker in selected_tickers}

        # Calculate weighting of share ownership as fraction of total sector value
        ticker_weightings = {ticker: ticker_value[ticker] / sum(ticker_value.values()) if sum(ticker_value.values()) > 0 else 0 for ticker in selected_tickers}

        # Calculate weighted change for each ticker
        weighted_change = 0
        for ticker in selected_tickers:
            if timestamp in ticker_data[ticker].index:
                # Get the close change for the ticker at the current timestamp
                close_change = ticker_data[ticker].loc[timestamp]['Close_change']
                weighted_change += close_change * ticker_weightings[ticker]

        sector_normalised.append(sector_normalised[-1] * (1 + weighted_change))

    # Get the data in a series with date in the index for plotting
    sector_normalised = pd.Series(sector_normalised[1:], index=ticker_data[selected_tickers[0]].index)

    return sector_normalised

def get_sector_plot(ticker_data: dict[str, pd.DataFrame]) -> list[float]:
    """
    Generate a list of weighted changes for each ticker in the sector based on order data.
    Args:
        ticker_data (dict[str, pd.DataFrame]): Dictionary where keys are sector names and values are DataFrames
            containing ticker price data with 'close_change' column.
    Returns:
        list[float]: List of weighted changes for each ticker in the sector.
    """
    selected_tickers = list(ticker_data.keys())

    sector_normalised = [100]

    # Ensure we are looking at the longest length of data to catch all historical data
    max_length_index = None
    max_length = 0

    for ticker in selected_tickers:
        length_data = len(ticker_data[ticker])
        if length_data > max_length:
            max_length = length_data
            max_length_index = ticker_data[ticker].index

    # Loop through and add weighting for each ticker in the sector
    for timestamp in max_length_index:

        # Get order data before the current date
        t = timestamp.date()

        # Calculate weighted change for each ticker
        num_tickers_trading_at_timestamp = len([ticker for ticker in selected_tickers if timestamp in ticker_data[ticker].index])
        weighted_change = 0
        for ticker in selected_tickers:
            if timestamp in ticker_data[ticker].index:
                # Get the close change for the ticker at the current timestamp
                close_change = ticker_data[ticker].loc[timestamp]['Close_change']
                weighted_change += close_change / num_tickers_trading_at_timestamp

        sector_normalised.append(sector_normalised[-1] * (1 + weighted_change))

    # Get the data in a series with date in the index for plotting
    sector_normalised = pd.Series(sector_normalised[1:], index=ticker_data[selected_tickers[0]].index)

    return sector_normalised

def get_value_plot(ticker_data: dict[str, pd.DataFrame], order_file: str) -> list[float]:
    """
    Generate a list of weighted changes for each ticker in the sector based on order data.
    Args:
        ticker_data (dict[str, pd.DataFrame]): Dictionary where keys are sector names and values are DataFrames
            containing ticker price data with 'close_change' column.
    Returns:
        list[float]: List of weighted changes for each ticker in the sector.
    """
    # Read the order data from CSV
    df_orders = pd.read_csv(order_file)
    df_orders['date'] = pd.to_datetime(df_orders['date'], format='%d/%m/%Y')

    selected_tickers = list(ticker_data.keys())

    selected_ticker_values = []

    # Ensure we are looking at the longest length of data to catch all historical data
    max_length_index = None
    max_length = 0

    for ticker in selected_tickers:
        length_data = len(ticker_data[ticker])
        if length_data > max_length:
            max_length = length_data
            max_length_index = ticker_data[ticker].index

    # Loop through and add weighting for each ticker in the sector
    for timestamp in max_length_index:

        # Get order data before the current date
        t = timestamp.date()
        current_datetime = datetime(t.year, t.month, t.day)
        current_order_data = df_orders[df_orders['date'] <= current_datetime]

        # Calculate the value of the shares owned for each ticker
        total_stock_value = 0
        for ticker in selected_tickers:
            if timestamp in ticker_data[ticker].index:
                # Get the value for the ticker at the current timestamp
                ticker_value = get_current_num_shares(current_order_data=current_order_data, ticker=ticker) * ticker_data[ticker].loc[timestamp]['Close']
                total_stock_value += ticker_value

        selected_ticker_values.append(total_stock_value)

    return selected_ticker_values

def get_stock_portfolio_value(current_order_data: pd.DataFrame, ticker: str) -> float:
    """
    Get the total value of the shares owned for a specific ticker using yfinance LastPrice.
    Returns:
        float: Total value ($USD) of the stocks owned for ticker.
    """
    num_shares = get_current_num_shares(current_order_data, ticker)
    total_value = num_shares * yf.Ticker(ticker).fast_info["lastPrice"]

    return total_value

def get_stock_portfolio_allocation(order_file: str) -> dict[str, float]:
    """
    Get the stock portfolio allocation from the orders CSV file.
    Returns:
        dict[str, float]: Dictionary where keys are tickers and values are their allocation percentages.
    """
    df_orders = pd.read_csv(order_file)

    tickers = df_orders['ticker'].unique()
    value_per_ticker = {ticker: get_stock_portfolio_value(df_orders, ticker) for ticker in tickers}

    allocations = {ticker: (value / sum(value_per_ticker.values()) * 100, value) if sum(value_per_ticker.values()) > 0 else 0 for ticker, value in value_per_ticker.items()}
    
    return allocations

def get_percentage_change_all_time(current_order_data: pd.DataFrame, ticker: str) -> float:
    """
    Get the percentage change of a stock based on the average price bought in for.
    Returns:
        float: Percentage change of the stock.
    """
    # Get all orders for the ticker
    orders = current_order_data[current_order_data['ticker'] == ticker]

    if orders.empty:
        return 0.0

    # Calculate the average price bought in
    total_cost = 0
    total_shares = 0

    for i in range(len(orders)):
        order = orders.iloc[i]
        if order["order_type"] != "sell":
            total_cost += order["num_shares"] * order["price"]
            total_shares += order["num_shares"]
        elif order["order_type"] == "sell":
            if total_shares == 0:
                raise ValueError("You can't sell shares you don't own!")
            avg_price = total_cost / total_shares
            total_cost -= order["num_shares"] * avg_price
            total_shares -= order["num_shares"]

    # Step 2: Calculate average cost of remaining shares
    if total_shares > 0:
        average_cost = total_cost / total_shares
    else:
        return 0.0

    # Get the current price using yfinance
    current_price = yf.Ticker(ticker).fast_info["lastPrice"]

    # Calculate percentage change
    unrealised_percentage_change = ((current_price - average_cost) / average_cost) * 100

    return unrealised_percentage_change

def get_top_and_bottom_3stocks_unrealized_changes_alltime(order_file: str):
    """
    Get the top 3 performing stocks of all time based on the Orders.csv file.
    Returns:
        dict[str, float]: Dictionary where keys are tickers and values are their percentage change based on the average price bought in for.
    """
    df_orders = pd.read_csv(order_file)

    tickers = df_orders['ticker'].unique()
    change_per_ticker = {ticker: get_percentage_change_all_time(df_orders, ticker) for ticker in tickers}

    # Sort the tickers by value in descending order and get the top 3
    sorted_tickers = sorted(change_per_ticker.items(), key=lambda x: x[1], reverse=True)
    top3_tickers = sorted_tickers[:3]
    bottom3_tickers = sorted_tickers[-3:]

    return {ticker: value for ticker, value in top3_tickers}, {ticker: value for ticker, value in bottom3_tickers}

def get_top_and_bottom_3subsectors_unrealized_changes_alltime_plot(order_file: str):
    """
    Get the top 3 performing subsectors of all time based on the Orders.csv file.
    Returns:
        dict[str, float]: Dictionary where keys are subsector names and values are their percentage change based on the average price bought in for.
    """
    df_orders = pd.read_csv(order_file)

    # Get unique subsectors
    subsectors = df_orders['Sector'].unique()
    
    change_per_subsector = {}
    
    for subsector in subsectors:
        tickers_in_subsector = df_orders[df_orders['subsector'] == subsector]['ticker'].unique()
        total_change = sum(get_percentage_change_all_time(df_orders, ticker) for ticker in tickers_in_subsector)
        change_per_subsector[subsector] = total_change / len(tickers_in_subsector) if tickers_in_subsector.size > 0 else 0

    # Sort the subsectors by value in descending order and get the top 3
    sorted_subsectors = sorted(change_per_subsector.items(), key=lambda x: x[1], reverse=True)
    top3_subsectors = sorted_subsectors[:3]
    bottom3_subsectors = sorted_subsectors[-3:]

    return {subsector: value for subsector, value in top3_subsectors}, {subsector: value for subsector, value in bottom3_subsectors}

if __name__ == "__main__":

    top3 = get_top_and_bottom_3stocks_unrealized_changes_alltime()

    print(top3)
    