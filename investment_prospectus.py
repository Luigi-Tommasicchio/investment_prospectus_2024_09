import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

np.random.seed(42)  

# Creating the tickers list:
tickers = ['NVO', 'ETSY', 'SQ', 'AVGO', 'ISRG', 'WDAY',
           'NFLX', 'NEE', 'AZN', 'AMZN', 'GOOG', 'ADSK', 'JPM', 'AMD', 'PYPL', 'NVDA',
           'NIO', 'MELI', 'SHOP', 'SPOT', 'TEAM', 'ENPH', 'TSLA',
           'SQM', 'ADBE', 'MA', 'TSM', 'AXON', 'CMG', 'DECK', 'TMUS',
           'ACN', 'IQV', 'CDNS', 'SNPS', 'EQIX', 'DLR', '^GSPC', 'SDG']

# Fetching the data from yahoo finance:
data = yf.download(tickers=tickers, start='2019-01-01', end='2024-01-01')['Adj Close']
data.info()

# Initial Plot of the data:
data.plot()
plt.show()

# Calculating the returns and removing the first line:
returns = data.pct_change().dropna()
returns.info()
returns.head()

# Number of assets:
n_assets = len(returns.columns) - 2 # The minus two is cause I wanna exclude the last 2 columns
n_portfolios = 10000                # of the df which cointain the benchmarks

# Calculate mean returns and covariance matrix:
mean_returns = returns.iloc[:, :n_assets].mean()
cov_matrix = returns.iloc[:, :n_assets].cov()

# Risk-free rate:
rf = 0.04  # Adjust to daily return (assuming 252 trading days per year)

# Initialize arrays to store portfolio metrics:
port_returns = []
port_volatility = []
sharpe_ratio = []
port_weights = []

# Generate random portfolios:
for i in range(n_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)  # Ensure that weights sum to 1
    
    # Calculate expected return and volatility:
    portfolio_return = np.sum(weights * mean_returns) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
    
    port_returns.append(portfolio_return)
    port_volatility.append(portfolio_volatility)
    sharpe_ratio.append((portfolio_return - rf) / portfolio_volatility)
    port_weights.append(weights)

# Convert lists to DataFrame for easy handling:
portfolios = pd.DataFrame({
    'Return': port_returns,
    'Volatility': port_volatility,
    'Sharpe Ratio': sharpe_ratio
})

# Add weights to the DataFrame:
for i, ticker in enumerate(returns.columns[:n_assets]):
    portfolios[ticker + ' Weight'] = [w[i] for w in port_weights]

# Plot the efficient frontier:
plt.figure(figsize=(10, 6))
plt.scatter(portfolios['Volatility'], portfolios['Return'], c=portfolios['Sharpe Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.show()

# Find the portfolio with the maximum Sharpe Ratio:
max_sharpe_idx = portfolios['Sharpe Ratio'].idxmax()
optimal_portfolio = portfolios.iloc[max_sharpe_idx]


# Extract and store tangency portfolio weights:
optimal_weights = optimal_portfolio[3:n_assets + 3]  # Skipping 'Return', 'Volatility', and 'Sharpe Ratio'
optimal_weights_dict = optimal_weights.to_dict()

# Save to CSV if needed:
optimal_weights.to_csv("optimal_portfolio_weights.csv", header=True)

# Output tangency portfolio stats:
print("Portfolio Weights:")
print(optimal_weights_dict)

# Align data and tangency portfolio weights if necessary
aligned_data = data.iloc[:, :n_assets]  # This selects the price data for the assets
aligned_weights = optimal_weights.values  # Convert weights to a NumPy array if they are in a Series format

# Compute the weighted sum of prices
optimal_portfolio_price = aligned_data.dot(aligned_weights)

# Normalize the prices to start at 100 for better comparison
optimal_portfolio_price_norm = optimal_portfolio_price / optimal_portfolio_price.iloc[0] * 100

# Extract the S&P 500 and SDG prices and normalize them as well
sp500_price = data['^GSPC']
sdg_price = data['SDG']

# Normalize SP500 and SDG to start at 100 for comparison
sp500_price_norm = sp500_price / sp500_price.iloc[0] * 100
sdg_price_norm = sdg_price / sdg_price.iloc[0] * 100

# Plot the prices of the Tangency Portfolio, S&P 500, and SDG
plt.figure(figsize=(10, 6))
plt.plot(optimal_portfolio_price_norm.index, optimal_portfolio_price_norm, label='Portfolio', color='blue')
plt.plot(sp500_price_norm.index, sp500_price_norm, label='S&P 500', color='fuchsia')
plt.plot(sdg_price_norm.index, sdg_price_norm, label='SDG', color='teal')
plt.legend()
plt.title('Tangency Portfolio Price vs S&P 500 and SDG')
plt.ylabel('Normalized Price (Starting at 100)')
plt.xlabel('Date')
plt.show()

optimal_portfolio_returns = optimal_portfolio_price.pct_change().dropna()

print(optimal_portfolio_returns)

sd_1y = optimal_portfolio_returns[-252:].std()
sd_3y = optimal_portfolio_returns[-756:].std()
sd_5y = optimal_portfolio_returns.std()

from sklearn.linear_model import LinearRegression

def calculate_beta_r2(portfolio_returns, benchmark_returns, window_size):
    """
    This function calculates beta and R-squared over a given window size.
    """
    portfolio_returns_window = portfolio_returns[-window_size:]
    benchmark_returns_window = benchmark_returns[-window_size:]

    # Reshape data for regression analysis
    X = benchmark_returns_window.values.reshape(-1, 1)  # SP500 returns
    y = portfolio_returns_window.values  # Portfolio returns

    # Perform linear regression
    reg = LinearRegression().fit(X, y)
    beta = reg.coef_[0]
    r_squared = reg.score(X, y)

    return beta, r_squared

# Define the window sizes for 1, 3, and 5 years
one_year_window = 252  # 1 year
three_year_window = 3 * 252  # 3 years
five_year_window = 1258  # 5 years

# Calculate daily returns of the S&P 500
sp500_returns = sp500_price.pct_change().dropna()

# Calculate beta and R-squared for 1, 3, and 5 years
beta_1yr, r2_1yr = calculate_beta_r2(optimal_portfolio_returns, sp500_returns, one_year_window)
beta_3yr, r2_3yr = calculate_beta_r2(optimal_portfolio_returns, sp500_returns, three_year_window)
beta_5yr, r2_5yr = calculate_beta_r2(optimal_portfolio_returns, sp500_returns, five_year_window)

# Display the results
print(f"1-Year Beta: {beta_1yr}, 1-Year R-squared: {r2_1yr}")
print(f"3-Year Beta: {beta_3yr}, 3-Year R-squared: {r2_3yr}")
print(f"5-Year Beta: {beta_5yr}, 5-Year R-squared: {r2_5yr}")

# Number of trading days in 1, 3, and max obs available in this case (almost 5y)
one_year_window = 252
three_year_window = 3 * 252
five_year_window = 1258

# Calculate standard deviation for each period (annualized)
std_1yr = np.std(optimal_portfolio_returns[-one_year_window:]) * np.sqrt(252)
std_3yr = np.std(optimal_portfolio_returns[-three_year_window:]) * np.sqrt(252)
std_5yr = np.std(optimal_portfolio_returns[-five_year_window:]) * np.sqrt(252)

# Create the table
metrics_data = {
    'Period': ['1-Year', '3-Year', '5-Year'],
    'Std Dev': [std_1yr, std_3yr, std_5yr],
    'Beta': [beta_1yr, beta_3yr, beta_5yr],
    'R-Squared': [r2_1yr, r2_3yr, r2_5yr]
}

# Convert to a DataFrame
metrics_table = pd.DataFrame(metrics_data)
print(metrics_table)
