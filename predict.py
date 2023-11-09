import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

def load_historical_prices(filename, split_factor=1.0):
    # Load historical stock prices from a CSV file using Pandas
    df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
    historical_prices = df['Close'].tolist()

    # Apply the split factor to adjust historical prices
    adjusted_prices = [price / split_factor for price in historical_prices]

    last_date = df.index[-1].date()  # Get the last date from the dataset
    return adjusted_prices, last_date

def plot_simulations(simulations, ticker):
    # Calculate the number of simulations
    num_simulations = len(simulations)

    # Create a gradient of colors from red to green
    colors = plt.cm.RdYlGn(np.linspace(0, 1, num_simulations))

    # Plot simulations with varying colors
    for i, sim in enumerate(simulations):
        date_range, prices = sim
        plt.plot(date_range, prices, color=colors[i])

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Monte Carlo Simulation: {ticker.upper()} Price Prediction')
    plt.show()


def run_monte_carlo_simulation(historical_prices, num_simulations, num_days, last_date):
    # Calculate daily returns
    returns = np.diff(historical_prices) / historical_prices[:-1]

    # Generate simulations
    simulations = []
    for i in range(num_simulations):
        prices = [historical_prices[-1]]
        current_date = last_date  # Set the current date to the last date
        date_range = [current_date]  # Initialize the date range with the start date
        for _ in range(num_days):
            daily_return = np.random.choice(returns)
            price = prices[-1] * (1 + daily_return)
            prices.append(price)
            current_date += pd.Timedelta(days=1)  # Increment the date by one day
            date_range.append(current_date)  # Append the new date

            # Check if the simulation exceeded 10,000% (100x) of the starting price
            if price > historical_prices[-1] * 75:
                break  # Exit the loop if exceeded

        # Only add simulations that didn't exceed 10,000%
        if prices[-1] <= historical_prices[-1] * 75:
            simulations.append((date_range, prices))

        # Print progress periodically
        if (i + 1) % 100 == 0:
            print(f'Simulations completed: {i + 1}/{num_simulations}')

    # Sort simulations by final price
    sorted_simulations = sorted(simulations, key=lambda x: x[1][-1])

    # Exclude both the highest and lowest three simulations
    num_to_exclude = int(num_simulations / 7)
    #return sorted_simulations[num_to_exclude:-num_to_exclude]
    return sorted_simulations[num_to_exclude:-num_to_exclude]

def find_average_prices(simulations, target_date):
    # Filter simulations for the target date
    target_simulations = [sim for sim in simulations if target_date in sim[0]]

    if not target_simulations:
        return None  # No simulations for the target date

    # Calculate average prices for the target date
    avg_prices = [sim[1][sim[0].index(target_date)] for sim in target_simulations]

    # Calculate the middle, low, and high averages
    middle_avg = np.median(avg_prices)
    low_avg = np.percentile(avg_prices, 25)  # 25th percentile
    high_avg = np.percentile(avg_prices, 75)  # 75th percentile

    # Calculate the overall average
    overall_avg = (middle_avg + low_avg + high_avg) / 3

    return middle_avg, low_avg, high_avg, overall_avg

def plot_simulations(simulations, ticker):
    # Calculate the number of simulations
    num_simulations = len(simulations)

    # Create a gradient of colors from red to green
    colors = plt.cm.RdYlGn(np.linspace(0, 1, num_simulations))

    # Plot simulations with varying colors
    for i, sim in enumerate(simulations):
        date_range, prices = sim
        plt.plot(date_range, prices, color=colors[i])

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Monte Carlo Simulation: {ticker.upper()} Price Prediction')
    plt.show()


def main():
    # Ask the user for the ticker symbol and split information
    ticker_input = input('Enter the ticker symbol (e.g., spy.us split 4/1): ')
    
    # Split the user input to get the ticker and split information
    ticker_parts = ticker_input.split(' split ')
    ticker = ticker_parts[0]
    
    # Check if split information is provided, default to 1.0 if not
    split_factor = 1.0
    if len(ticker_parts) == 2:
        split_info = ticker_parts[1]
        split_values = split_info.split('/')
        if len(split_values) == 2:
            try:
                numerator = float(split_values[0])
                denominator = float(split_values[1])
                split_factor = numerator / denominator
            except ValueError:
                pass

    # Try to find the CSV file in 'Stocks' and 'ETFs' folders
    stock_filename = os.path.join('Stocks', f'{ticker}.txt')
    etf_filename = os.path.join('ETFs', f'{ticker}.txt')

    if os.path.exists(stock_filename):
        historical_prices, last_date = load_historical_prices(stock_filename, split_factor)
    elif os.path.exists(etf_filename):
        historical_prices, last_date = load_historical_prices(etf_filename, split_factor)
    else:
        print(f'Ticker {ticker} not found in Stocks or ETFs folders.')
        return

    num_simulations = 2000
    num_days = 2250

    simulations = run_monte_carlo_simulation(historical_prices, num_simulations, num_days, last_date)
    
    while True:
        command = input('Enter a command (e.g., "findavg 10-10-2023", "showplot", "exit"): ')
        
        if command.lower() == 'exit':
            break
        elif command.startswith('findavg '):
            target_date_str = command.split(' ')[1]
            target_date = pd.to_datetime(target_date_str, format='%m-%d-%Y').date()
            avg_prices = find_average_prices(simulations, target_date)
            if avg_prices is not None:
                middle_avg, low_avg, high_avg, overall_avg = avg_prices
                print(f'The middle average for {ticker} on {target_date} is: ${middle_avg:.2f}')
                print(f'The low average for {ticker} on {target_date} is: ${low_avg:.2f}')
                print(f'The high average for {ticker} on {target_date} is: ${high_avg:.2f}')
                print(f'The overall average for {ticker} on {target_date} is: ${overall_avg:.2f}')
            else:
                print(f'No simulations available for {target_date}')
        elif command.lower() == 'showplot':
            plot_simulations(simulations, ticker)

if __name__ == '__main__':
    main()
