from langflow.custom import Component
from langflow.inputs import StrInput, FloatInput
from langflow.template import Output
from langflow.field_typing import Text
from typing import Callable
from typing import Tuple
import sys
import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import expected_returns, efficient_frontier, plotting, objective_functions
from yahoo_fin import stock_info
import requests
import cvxpy as cp
from loguru import logger
from pathlib import Path


# Optional: Adjust log file path
log_file_path = Path("./logs/langflow_component.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True)

# Remove default handlers (only if not handled globally)
logger.remove()

# Add back handlers: one for terminal, one for file
logger.add(sys.stdout, level="DEBUG")  # for container logs
logger.add(log_file_path, level="DEBUG", rotation="500 KB", retention="5 days")

class InvestmentPortfolioOptimizer(Component):
    display_name = "Investment Portfolio Optimizer"
    description = "Optimizes a stock portfolio using Efficient Frontier and returns key performance metrics."
    
    inputs = [
        MessageTextInput(
            name="investment_json",
            display_name="Investment JSON",
            required=True,
            tool_mode=True,
        ),
    ]

    outputs = [
            Output(name="portfolio_generation_status", display_name="Portfolio Generation Status", method="execute"),
    ]

    
    def get_and_print_nifty50_tickers(self):
        """
        Fetches the list of stock tickers in the Nifty 50 index, prints them, and returns the list.

        Returns:
            list: A list of tickers in the Nifty 50 index.
        """
        nifty50_tickers = stock_info.tickers_nifty50()

        # Print the tickers inside the method
        logger.debug('Nifty 50 Tickers: ', nifty50_tickers)

        # Return the list of tickers as assets
        assets = nifty50_tickers
        return assets

    def get_historical_data(self, assets):
        """
        Fetches historical closing price data for the given assets (stocks),
        processes the data by replacing Inf values with NaN, and removes columns with NaN values.

        Args:
            assets (list): List of stock tickers to fetch data for.

        Returns:
            DataFrame: Processed historical closing price data for the assets.
        """
        
        os.environ['TZ'] = 'Asia/Kolkata'
        
        # Get the historical price data
        data = yf.download(assets, start="2015-04-01", end="2025-04-01")

        # 'Close' indicating closing price of the stocks are required
        data = data['Close']

        # Replace Inf values with NaN and then drop the columns with any NaN values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis=1, how='any', inplace=True)  # Drop columns with NaN or Inf values

        return data

    def plot_historical_prices(self, data, file_name='./Historical_Closing_Prices.png'):
        """
        Plots the historical adjusted close prices for all assets and saves the plot to a file.

        Args:
            data (DataFrame): A DataFrame with the historical price data for the assets.
            file_name (str): File path to save the plot image.

        Saves:
            A plot of historical closing prices saved as a PNG file.
        """
        # Plot the historical adjusted close prices of all assets
        plt.figure(figsize=(12, 6))  # Slightly wider for more space
        
        data.index = pd.to_datetime(data.index)

        for ticker in data.columns:
            plt.plot(data.index, data[ticker], label=ticker)
            
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.title("Historical Adjusted Close Prices")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")

        # Add legend outside plot
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize='small', ncol=2)
        plt.grid(True)

        # Save with bbox_inches='tight' to prevent clipping
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    def mu_sigma(self, data):
        """
        Calculates the expected returns and the covariance matrix for the given asset data.

        Args:
            data (DataFrame): A DataFrame containing historical price data for the assets.

        Returns:
            tuple: A tuple containing the expected returns and the covariance matrix.
                - returns_expected_value (Series): The expected returns for each asset.
                - cov_matrix (DataFrame): The covariance matrix of the asset returns.
        """
        # Calculate expected returns and covariance matrix
        returns_expected_value = expected_returns.mean_historical_return(data)
        cov_matrix = risk_models.sample_cov(data)

        return returns_expected_value, cov_matrix

    def get_efficient_frontier(self, data, returns_expected_value, cov_matrix, user_criteria='max_sharpe'):
        """
        Optimizes the Efficient Frontier based on the specified user criteria.

        Args:
            user_criteria (str): The optimization criterion ('max_sharpe', 'target_return', or 'min_risk').
            data (DataFrame): The historical price data for the assets, used to calculate expected returns and covariance.

        Returns:
            ef (EfficientFrontier): The optimized Efficient Frontier object.
        """

        # Initialize Efficient Frontier optimizer
        ef = None

        # Efficient Frontier Optimization based on user criteria
        if user_criteria == 'max_sharpe':
            ef = EfficientFrontier(returns_expected_value, cov_matrix, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg, gamma=1)  # Regularization term for max Sharpe
        elif user_criteria == 'target_return':
            ef = EfficientFrontier(returns_expected_value, cov_matrix, weight_bounds=(-1, 1))
        elif user_criteria == 'min_risk':
            ef = EfficientFrontier(returns_expected_value, cov_matrix, weight_bounds=(0, 1))

        return ef

    def save_efficient_frontier_plot(self, ef_opt, file_name='./Efficient_Frontier.png'):
        """
        Generates and saves the efficient frontier plot to a file.

        Args:
            ef_opt (EfficientFrontier): An instance of the EfficientFrontier class containing optimization results.
            file_name (str): The name of the file where the plot will be saved (default is 'efficient_frontier.png').

        Returns:
            None
        """
        # Plot the Efficient Frontier
        plotting.plot_efficient_frontier(ef_opt, show_assets=True)

        # Save the plot to the specified file
        plt.savefig(file_name)  # Save as PNG or other formats (e.g., .jpg, .pdf)
        plt.close()  # Close the plot to avoid memory issues

    def optimize_portfolio(self, ef, risk_free_rate=0.06, target_return=0.5, user_criteria='max_sharpe'):
        """
        Optimizes the portfolio based on the specified user criteria and returns the weights.

        Args:
            user_criteria (str): The optimization criterion ('max_sharpe', 'target_return', or 'min_risk').
            ef (EfficientFrontier): An instance of the EfficientFrontier class.
            risk_free_rate (float, optional): The risk-free rate, required if user_criteria is 'max_sharpe'.
            target_return (float, optional): The target return, required if user_criteria is 'target_return'.

        Returns:
            dict: The clean portfolio weights after optimization.
        """
        if user_criteria == 'max_sharpe':
            # Maximize Sharpe ratio
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif user_criteria == 'target_return':
            # Efficient frontier for a specific target return with market neutral constraint
            ef.efficient_return(target_return=target_return, market_neutral=True)
        elif user_criteria == 'min_risk':
            # Minimize risk (volatility)
            ef.min_volatility()

        # Clean the weights and return
        weights = ef.clean_weights()
        print('Weights: ', weights)
        return weights


    def calculate_performance(self, ef, risk_free_rate=0.06, file_name='./data/Portfolio_Performance.csv'):
        """
        Calculates the expected performance of the portfolio, including expected return,
        volatility, and Sharpe ratio. Also writes the values to a CSV file.
    
        Args:
            ef (EfficientFrontier): An instance of the EfficientFrontier class.
            risk_free_rate (float): The risk-free rate used to calculate the Sharpe ratio.
            file_name (str): The path to the CSV file to save the performance data.
    
        Returns:
            tuple: A tuple containing expected annual return, annual volatility, and Sharpe ratio.
        """
        # Get the expected performance (returns and volatility)
        performance = ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)

        # Extract the values
        expected_annual_return = performance[0]
        annual_volatility = performance[1]
        sharpe_ratio = performance[2]

        # Create a DataFrame with one row and three columns
        performance_df = pd.DataFrame([{
            "Expected_Annual_Return": expected_annual_return,
            "Annual_Volatility": annual_volatility,
            "Sharpe_Ratio": sharpe_ratio
        }])

        # Save to CSV
        performance_df.to_csv(file_name, index=False)

        return expected_annual_return, annual_volatility, sharpe_ratio


    def generate_portfolio_display(self, weights, total_amount, file_name='./Portfolio_Display_Data.csv'):
        """
        Generates a portfolio display with details about each stock, including:
        - Company name
        - Market sector
        - Percentage allocation
        - Weighted amount (based on allocation)
        - Previous day's closing price

        Args:
            weights (dict): A dictionary where keys are stock tickers and values are the corresponding weights.
            total_amount (float): The total amount to be invested in the portfolio.

        Returns:
            pd.DataFrame: A DataFrame containing portfolio details for each stock.
        """
        
        os.environ['TZ'] = 'Asia/Kolkata'
        
        portfolio_display_list = []

        for ticker, weight in weights.items():
            stock = yf.Ticker(ticker)

            # Get the latest stock price
            stock_price = stock.history(interval="1h", period="1d")['Close'].iloc[-1]

            # Get company info
            stock_info = stock.info
            company_name = stock_info.get('shortName', 'N/A')
            market_sector = stock_info.get('sector', 'N/A')

            # Add stock information to the portfolio display list
            portfolio_display_list.append({
                'Company_Name': company_name,
                'Market_Sector': market_sector,
                'Percentage_Allocation': weight,
                'Weighted_Amount': weight * total_amount,
                'Prev_Day_Closing': stock_price
            })

        # Convert the portfolio display list to a DataFrame
        portfolio_display_data = pd.DataFrame(portfolio_display_list)

        print(portfolio_display_data)

        # Save the DataFrame to a CSV file
        portfolio_display_data.to_csv(file_name, index=False)

        return portfolio_display_data
        
        
    
    def execute(self) -> Message: 
        
        logger.debug("Running portfolio optimization...")
       
        # Parse input JSON string
        try:
            investment_data = json.loads(self.investment_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse input JSON: {e}")
            return f"Error parsing input: {e}"

        # Extract parameters
        user_criteria = investment_data.get("user_criteria", "max_sharpe")
        total_amount = float(investment_data.get("total_amount", 100000))
        target_return = float(investment_data.get("target_return", 0.0))
        
        risk_free_rate = 0.06
        
        assets = self.get_and_print_nifty50_tickers()
        data = self.get_historical_data(assets)
        #self.plot_historical_prices(data, './data/Historical_Closing_Prices.png')
        returns_expected_value, cov_matrix = self.mu_sigma(data)
        ef = self.get_efficient_frontier(data, returns_expected_value, cov_matrix, user_criteria)
        #ef_opt = self.get_efficient_frontier(data, returns_expected_value, cov_matrix, user_criteria)
        #self.save_efficient_frontier_plot(ef_opt, './data/Efficient_Frontier.png')
        weights = self.optimize_portfolio(ef, risk_free_rate, target_return, user_criteria)
        expected_annual_return, annual_volatility, sharpe_ratio = self.calculate_performance(ef, risk_free_rate, file_name='./data/Portfolio_Performance.csv')
        portfolio_data = self.generate_portfolio_display(weights, total_amount, './data/Portfolio_Display_Data.csv')
        
        portfolio_dashboard_link = "http://localhost:8000/Portfolio_Dashboard.pbix"
        
        target_return_msg1 = f"- As per your requirements, we are offering a market neutral portfolio designed to deliver stable, risk-adjusted returns by using both long and short positions.\n\n"
        target_return_msg2 = f"- Please revisit us periodically for portfolio rebalancing to maintain optimal performance and neutrality.\n\n"
        
        optimization_criteria_dict = {'target_return': 'Target Return', 'max_sharpe': 'Maximum Sharpe Ratio', 'min_risk': 'Minimum Volatility'}
        
        # Format the summary message
        summary = (
            f"📈 **Investment Portfolio Summary**\n\n"
            f"- **Optimization Criteria**: {optimization_criteria_dict.get(user_criteria)}\n"
            f"- **Expected Annual Return**: {expected_annual_return:.2%}\n"
            f"- **Annual Volatility (Risk)**: {annual_volatility:.2%}\n"
            f"- **Sharpe Ratio**: {sharpe_ratio:.2f}\n"
        )
        

        # Format portfolio_data as table
        try:
            portfolio_table = portfolio_data.to_markdown(index=False)
        except:
            portfolio_table = portfolio_data.to_string(index=False)

        if user_criteria == 'target_return': 
            chat_output = f"{target_return_msg1}\n\n{target_return_msg2}\n\n{summary}\n\n📊 **Portfolio Details:**\n\n{portfolio_table}\n\n"
        else: 
            chat_output = f"{summary}\n\n📊 **Portfolio Details:**\n\n{portfolio_table}\n\n"
        
        chat_output += f"- Download 🔗 [Portfolio Dashboard (Power BI)]({portfolio_dashboard_link}). (Please refresh data in your .pbix file.)"
        
        return chat_output
        
        
