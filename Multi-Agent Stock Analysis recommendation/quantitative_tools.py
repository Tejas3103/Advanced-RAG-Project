"""
Quantitative tools for stock analysis using yfinance and external APIs.
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import numpy as np
from functools import lru_cache
from langchain.tools import tool

@tool("get_stock_prices")
def get_stock_prices(ticker: str, period: str = "1y") -> str:
    """Get historical price data for a stock ticker.
    
    Args:
        ticker: The stock symbol (e.g., AAPL, MSFT)
        period: Time period for data retrieval (e.g., 1d, 1mo, 1y)
        
    Returns:
        Summary of retrieved data or error message
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return f"No data found for ticker {ticker}."
        
        return f"Retrieved {len(hist)} days of price data for {ticker}. Latest close: ${hist['Close'].iloc[-1]:.2f}"
    except Exception as e:
        return f"Error retrieving stock data for {ticker}: {str(e)}"

@tool("get_price_extremes")
def get_price_extremes(ticker: str, period: str = "1y") -> str:
    """Find the highest and lowest prices with dates for a stock ticker.
    
    Args:
        ticker: The stock symbol (e.g., AAPL, MSFT)
        period: Time period for data retrieval (e.g., 1d, 1mo, 1y)
        
    Returns:
        Formatted list of 5 highest and 5 lowest prices with dates
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return f"No data found for ticker {ticker}."
        
        # Sort by closing price
        sorted_closes = hist['Close'].sort_values()
        
        # Get 5 lowest and highest
        lowest = sorted_closes.head(5)
        highest = sorted_closes.tail(5)
        
        result = "Price extremes:\n\n"
        result += "Lowest Prices:\n"
        for date, price in lowest.items():
            result += f"- {date.strftime('%Y-%m-%d')}: ${price:.2f}\n"
        
        result += "\nHighest Prices:\n"
        for date, price in highest.items():
            result += f"- {date.strftime('%Y-%m-%d')}: ${price:.2f}\n"
        
        return result
    except Exception as e:
        return f"Error analyzing price extremes for {ticker}: {str(e)}"

@tool("analyze_recent_trend")
def analyze_recent_trend(ticker: str, days: int = 30) -> str:
    """Analyze the recent price trend for a stock ticker.
    
    Args:
        ticker: The stock symbol (e.g., AAPL, MSFT)
        days: Number of days to analyze (default: 30)
        
    Returns:
        Analysis of recent price trend including percent change and assessment
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            return f"No data found for ticker {ticker}."
        
        # Get recent data
        recent = hist.tail(days)
        if len(recent) < days:
            return f"Not enough data for {ticker}. Only {len(recent)} days available."
        
        # Calculate metrics
        start_price = recent['Close'].iloc[0]
        end_price = recent['Close'].iloc[-1]
        pct_change = ((end_price - start_price) / start_price) * 100
        
        # Determine trend
        if pct_change > 5:
            trend = "strongly bullish"
        elif pct_change > 0:
            trend = "mildly bullish"
        elif pct_change > -5:
            trend = "mildly bearish"
        else:
            trend = "strongly bearish"
        
        result = f"Trend analysis for the last {days} days:\n"
        result += f"- Period: {recent.index[0].strftime('%Y-%m-%d')} to {recent.index[-1].strftime('%Y-%m-%d')}\n"
        result += f"- Starting price: ${start_price:.2f}\n"
        result += f"- Ending price: ${end_price:.2f}\n"
        result += f"- Percent change: {pct_change:.2f}%\n"
        result += f"- Trend assessment: {trend}\n"
        
        return result
    except Exception as e:
        return f"Error analyzing recent trend for {ticker}: {str(e)}"

@tool("analyze_portfolio_risk")
def analyze_portfolio_risk(portfolio_text: str, risk_free_rate: float = 0.04) -> str:
    """Analyze risk and return characteristics of a portfolio using CAPM.
    
    Args:
        portfolio_text: Description of portfolio in format "X% in TICKER1, Y% in TICKER2"
        risk_free_rate: Annual risk-free rate (default: 4%)
        
    Returns:
        Portfolio risk metrics including beta, alpha, Sharpe ratio
    """
    try:
        # Extract tickers and weights using simple parsing
        portfolio_items = []
        for item in portfolio_text.split(','):
            parts = item.strip().split('in')
            if len(parts) == 2:
                try:
                    weight = float(parts[0].strip().replace('%', '')) / 100
                    ticker = parts[1].strip()
                    portfolio_items.append((ticker, weight))
                except ValueError:
                    continue
        
        if not portfolio_items:
            return "Could not parse portfolio. Please use format like '30% in AAPL, 70% in MSFT'"
        
        # Detect market region and select appropriate benchmark
        is_indian_market = any(".NS" in ticker or ".BO" in ticker for ticker, _ in portfolio_items)
        is_european_market = any(ticker.split('.')[-1] in ["DE", "PA", "L", "MC", "AS"] for ticker, _ in portfolio_items if "." in ticker)
        
        # Select appropriate market index based on portfolio composition
        if is_indian_market:
            market_ticker = "^NSEI"  # NIFTY 50 index
        elif is_european_market:
            market_ticker = "^STOXX50E"  # EURO STOXX 50
        else:
            market_ticker = "SPY"  # Default to S&P 500 ETF
        
        # Get market index data
        market = yf.Ticker(market_ticker)
        market_hist = market.history(period="1y")["Close"]
        if market_hist.empty:
            # Fallback to SPY if the selected index doesn't have data
            market = yf.Ticker("SPY")
            market_hist = market.history(period="1y")["Close"]
        
        market_returns = market_hist.pct_change().dropna()
        
        # Get data for all tickers
        portfolio_data = {}
        ticker_errors = []
        
        for ticker, weight in portfolio_items:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")["Close"]
                
                if hist.empty or len(hist) < 20:  # Require at least 20 days of data
                    ticker_errors.append(f"Insufficient price history for {ticker}")
                    continue
                
                returns = hist.pct_change().dropna()
                portfolio_data[ticker] = {
                    'returns': returns,
                    'weight': weight,
                    'current_price': hist.iloc[-1]
                }
            except Exception as e:
                ticker_errors.append(f"Error processing {ticker}: {str(e)}")
        
        if not portfolio_data:
            if ticker_errors:
                return "Could not analyze portfolio: " + "; ".join(ticker_errors)
            return "Could not retrieve data for any tickers in the portfolio."
        
        # Calculate individual stock metrics first (without requiring market alignment)
        result = f"## Portfolio Analysis\n\n"
        result += f"### Portfolio Composition\n"
        for ticker, data in portfolio_data.items():
            result += f"- {ticker}: {data['weight'] * 100:.1f}% (Current price: {data['current_price']:.2f})\n"
        
        # Calculate volatility for each stock and portfolio
        volatilities = {}
        for ticker, data in portfolio_data.items():
            volatilities[ticker] = data['returns'].std() * np.sqrt(252)  # Annualized
        
        # Calculate portfolio return and volatility
        # For simplicity, use average return without market alignment
        avg_returns = {}
        for ticker, data in portfolio_data.items():
            avg_returns[ticker] = data['returns'].mean() * 252  # Annualized
        
        portfolio_return = sum(avg_returns[ticker] * data['weight'] for ticker, data in portfolio_data.items())
        
        # Calculate portfolio volatility (simplified without covariance matrix)
        portfolio_volatility = sum(volatilities[ticker] * data['weight'] for ticker, data in portfolio_data.items())
        
        # Try to calculate beta if possible
        betas = {}
        has_beta = False
        try:
            # Try to find overlapping dates for beta calculation
            all_dates = set()
            for ticker, data in portfolio_data.items():
                all_dates.update(data['returns'].index)
            
            common_dates = all_dates.intersection(market_returns.index)
            
            if len(common_dates) >= 20:  # Require at least 20 common trading days
                has_beta = True
                aligned_market_returns = market_returns.loc[common_dates].sort_index()
                
                for ticker, data in portfolio_data.items():
                    aligned_returns = data['returns'].loc[data['returns'].index.intersection(common_dates)].sort_index()
                    if len(aligned_returns) >= 20:
                        # Ensure aligned market returns match the stock returns
                        matching_market = aligned_market_returns.loc[aligned_returns.index]
                        
                        # Calculate beta
                        covariance = np.cov(aligned_returns, matching_market)[0, 1]
                        market_variance = np.var(matching_market)
                        beta = covariance / market_variance if market_variance != 0 else 1.0
                        betas[ticker] = beta
        except Exception as e:
            # If beta calculation fails, continue without it
            pass
        
        # Calculate portfolio metrics
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
        
        # Format results
        result += f"\n### Overall Portfolio Metrics\n"
        result += f"- Estimated Annual Return: {portfolio_return * 100:.2f}%\n"
        result += f"- Portfolio Volatility: {portfolio_volatility * 100:.2f}%\n"
        result += f"- Sharpe Ratio: {sharpe_ratio:.2f}\n"
        
        if has_beta and betas:
            portfolio_beta = sum(betas.get(ticker, 1.0) * data['weight'] for ticker, data in portfolio_data.items())
            result += f"- Portfolio Beta: {portfolio_beta:.2f}\n"
            
            # Add beta for individual stocks
            result += f"\n### Individual Asset Betas\n"
            for ticker in portfolio_data:
                if ticker in betas:
                    result += f"- {ticker} Beta: {betas[ticker]:.2f}\n"
                else:
                    result += f"- {ticker} Beta: Not available\n"
        else:
            result += "\nNote: Beta calculation not available due to insufficient market data overlap.\n"
        
        result += f"\n### Individual Asset Volatilities\n"
        for ticker in portfolio_data:
            result += f"- {ticker} Volatility: {volatilities[ticker] * 100:.2f}%\n"
        
        # Add warning about any tickers with errors
        if ticker_errors:
            result += f"\n### Warning\n"
            for error in ticker_errors:
                result += f"- {error}\n"
        
        return result
    except Exception as e:
        return f"Error analyzing portfolio: {str(e)}"

@tool("get_financial_ratios")
def get_financial_ratios(ticker: str) -> str:
    """Get key financial ratios for a company using yfinance.
    
    Args:
        ticker: The stock symbol (e.g., AAPL, MSFT)
        
    Returns:
        Key financial ratios including valuation, profitability, and growth
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get key statistics and financials
        info = stock.info
        
        result = f"## Financial Ratios for {ticker}\n\n"
        
        # Valuation ratios
        result += "### Valuation Ratios\n"
        ratios = [
            ("P/E Ratio (TTM)", "trailingPE"),
            ("Forward P/E", "forwardPE"),
            ("Price/Sales (TTM)", "priceToSalesTrailing12Months"),
            ("Price/Book", "priceToBook"),
            ("Enterprise Value/EBITDA", "enterpriseToEbitda"),
        ]
        
        for label, key in ratios:
            value = info.get(key, None)
            if value:
                result += f"- {label}: {value:.2f}\n"
            else:
                result += f"- {label}: N/A\n"
        
        # Profitability ratios
        result += "\n### Profitability Metrics\n"
        profitability = [
            ("Profit Margin", "profitMargins"),
            ("Operating Margin (TTM)", "operatingMargins"),
            ("Return on Assets (TTM)", "returnOnAssets"),
            ("Return on Equity (TTM)", "returnOnEquity"),
        ]
        
        for label, key in profitability:
            value = info.get(key, None)
            if value:
                result += f"- {label}: {value*100:.2f}%\n"
            else:
                result += f"- {label}: N/A\n"
        
        # Growth metrics
        result += "\n### Growth Metrics\n"
        growth = [
            ("Revenue Growth (YoY)", "revenueGrowth"),
            ("Earnings Growth (YoY)", "earningsGrowth"),
        ]
        
        for label, key in growth:
            value = info.get(key, None)
            if value:
                result += f"- {label}: {value*100:.2f}%\n"
            else:
                result += f"- {label}: N/A\n"
        
        # Dividend information
        result += "\n### Dividend Information\n"
        dividend_info = [
            ("Dividend Yield", "dividendYield"),
            ("Dividend Rate", "dividendRate"),
            ("Payout Ratio", "payoutRatio"),
        ]
        
        for label, key in dividend_info:
            value = info.get(key, None)
            if value and label == "Dividend Yield":
                result += f"- {label}: {value*100:.2f}%\n"
            elif value:
                result += f"- {label}: {value:.2f}\n"
            else:
                result += f"- {label}: N/A\n"
        
        return result
    except Exception as e:
        return f"Error retrieving financial ratios for {ticker}: {str(e)}"

@lru_cache(maxsize=100)
def _get_finnhub_data(endpoint, params):
    """Helper function to make Finnhub API calls with caching."""
    base_url = "https://finnhub.io/api/v1"
    params['token'] = "d0ekrjpr01qkbclc0kb0d0ekrjpr01qkbclc0kbg"
    response = requests.get(f"{base_url}/{endpoint}", params=params)
    return response.json()

@tool("get_earnings_calendar")
def get_earnings_calendar(ticker: str) -> str:
    """Get upcoming and past earnings dates, estimates, and actual results.
    
    Args:
        ticker: The stock symbol (e.g., AAPL, MSFT, RELIANCE.NS)
        
    Returns:
        Earnings surprise history and upcoming announcements
    """
    try:
        # Clean ticker for Finnhub compatibility
        clean_ticker = ticker.split('.')[0] if '.' in ticker else ticker
        
        # Get earnings surprises (past earnings)
        surprise_data = _get_finnhub_data("stock/earnings", {"symbol": clean_ticker})
        
        # Get company earnings calendar (upcoming earnings)
        today = datetime.now()
        from_date = today.strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=90)).strftime("%Y-%m-%d")
        calendar_data = _get_finnhub_data("calendar/earnings", {
            "symbol": clean_ticker,
            "from": from_date,
            "to": to_date
        })
        
        result = f"## Earnings Information for {ticker}\n\n"
        
        # Check if we have valid data
        if not surprise_data or len(surprise_data) == 0:
            result += "⚠️ No historical earnings data available for this ticker in Finnhub.\n"
            if '.' in ticker:
                result += f"Note: For international stocks like {ticker}, earnings data may be limited. Tried searching with ticker '{clean_ticker}'.\n\n"
            
            # Fallback to yfinance for basic earnings information
            try:
                stock = yf.Ticker(ticker)
                calendar = stock.calendar
                if calendar is not None and not calendar.empty:
                    result += "### Upcoming Earnings (from yfinance)\n"
                    for event_type, date in calendar.items():
                        if 'Earnings' in str(event_type):
                            result += f"- {event_type}: {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date}\n"
            except:
                pass
        else:
            # Format past earnings surprises
            result += "### Historical Earnings Surprises\n"
            result += "| Quarter | Date | Estimated EPS | Actual EPS | Surprise % |\n"
            result += "|---------|------|--------------|------------|------------|\n"
            
            for quarter in surprise_data:
                period = quarter.get('period', 'N/A')
                date = quarter.get('period', 'N/A')
                est_eps = quarter.get('estimate', 'N/A')
                act_eps = quarter.get('actual', 'N/A')
                surprise_pct = quarter.get('surprisePercent', 'N/A')
                
                if est_eps != 'N/A' and act_eps != 'N/A':
                    result += f"| {period} | {date} | ${est_eps:.2f} | ${act_eps:.2f} | {surprise_pct:.2f}% |\n"
        
        # Format upcoming earnings
        result += "\n### Upcoming Earnings Announcements\n"
        if calendar_data and 'earningsCalendar' in calendar_data and len(calendar_data['earningsCalendar']) > 0:
            for event in calendar_data['earningsCalendar']:
                date = event.get('date', 'N/A')
                time = event.get('hour', 'N/A')
                est_eps = event.get('epsEstimate', 'N/A')
                
                result += f"- **Next Earnings Date**: {date}\n"
                if time != 'N/A':
                    result += f"- **Time**: {time}\n"
                if est_eps != 'N/A':
                    result += f"- **EPS Estimate**: ${est_eps:.2f}\n"
        else:
            result += "No upcoming earnings announcements found.\n"
            
            # Try yfinance as fallback
            try:
                stock = yf.Ticker(ticker)
                next_earnings = stock.calendar
                if next_earnings is not None and not next_earnings.empty:
                    result += "\nFallback data from yfinance:\n"
                    for event_type, date in next_earnings.items():
                        if 'Earnings' in str(event_type):
                            result += f"- {event_type}: {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date}\n"
            except:
                pass
        
        return result
    except Exception as e:
        return f"Error retrieving earnings data for {ticker}: {str(e)}"

@tool("get_analyst_recommendations")
def get_analyst_recommendations(ticker: str) -> str:
    """Get current analyst recommendations and price targets.
    
    Args:
        ticker: The stock symbol (e.g., AAPL, MSFT, RELIANCE.NS)
        
    Returns:
        Buy/sell/hold consensus and price target analysis
    """
    try:
        # Clean ticker for Finnhub compatibility
        clean_ticker = ticker.split('.')[0] if '.' in ticker else ticker
        
        # Get recommendation trends
        recommendation_data = _get_finnhub_data("stock/recommendation", {"symbol": clean_ticker})
        
        # Get price target
        price_target_data = _get_finnhub_data("stock/price-target", {"symbol": clean_ticker})
        
        result = f"## Analyst Recommendations for {ticker}\n\n"
        
        has_finnhub_data = False
        
        # Process price target data
        result += "### Price Target\n"
        if price_target_data and price_target_data.get('targetMean') is not None:
            has_finnhub_data = True
            last_updated = price_target_data.get('lastUpdated', 'N/A')
            target_high = price_target_data.get('targetHigh', 'N/A')
            target_low = price_target_data.get('targetLow', 'N/A')
            target_mean = price_target_data.get('targetMean', 'N/A')
            target_median = price_target_data.get('targetMedian', 'N/A')
            
            result += f"- **Last Updated**: {last_updated}\n"
            result += f"- **Target High**: ${target_high:.2f}\n" if target_high != 'N/A' else "- **Target High**: N/A\n"
            result += f"- **Target Low**: ${target_low:.2f}\n" if target_low != 'N/A' else "- **Target Low**: N/A\n"
            result += f"- **Target Mean**: ${target_mean:.2f}\n" if target_mean != 'N/A' else "- **Target Mean**: N/A\n"
            result += f"- **Target Median**: ${target_median:.2f}\n" if target_median != 'N/A' else "- **Target Median**: N/A\n"
        else:
            result += "No price target data available from Finnhub.\n"
        
        # Process recommendation trends
        result += "\n### Recommendation Trends\n"
        if recommendation_data and len(recommendation_data) > 0:
            has_finnhub_data = True
            # Get the most recent recommendation
            latest = recommendation_data[0]
            period = latest.get('period', 'N/A')
            
            result += f"**Latest Recommendations ({period}):**\n"
            result += f"- Strong Buy: {latest.get('strongBuy', 0)}\n"
            result += f"- Buy: {latest.get('buy', 0)}\n"
            result += f"- Hold: {latest.get('hold', 0)}\n"
            result += f"- Sell: {latest.get('sell', 0)}\n"
            result += f"- Strong Sell: {latest.get('strongSell', 0)}\n"
            
            # Calculate consensus
            total = sum([
                latest.get('strongBuy', 0),
                latest.get('buy', 0),
                latest.get('hold', 0),
                latest.get('sell', 0),
                latest.get('strongSell', 0)
            ])
            
            if total > 0:
                buy_weight = (latest.get('strongBuy', 0) * 2 + latest.get('buy', 0)) / (total * 2)
                sell_weight = (latest.get('strongSell', 0) * 2 + latest.get('sell', 0)) / (total * 2)
                hold_weight = latest.get('hold', 0) / total
                
                # Determine consensus rating
                if buy_weight > 0.7:
                    consensus = "STRONG BUY"
                elif buy_weight > 0.4:
                    consensus = "BUY"
                elif sell_weight > 0.7:
                    consensus = "STRONG SELL"
                elif sell_weight > 0.4:
                    consensus = "SELL"
                else:
                    consensus = "HOLD"
                
                result += f"\n**Overall Consensus**: {consensus}\n"
        else:
            result += "No recommendation data available from Finnhub.\n"
        
        # For international stocks, provide a fallback using yfinance
        if not has_finnhub_data:
            result += "\n### Alternative Data from Yahoo Finance\n"
            if '.' in ticker:
                result += f"Note: For international stocks like {ticker}, analyst data from Finnhub may be limited. Tried searching with ticker '{clean_ticker}'.\n\n"
            
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Get analyst recommendations
                if 'recommendationMean' in info:
                    rec_mean = info['recommendationMean']
                    result += f"- **Recommendation Mean**: {rec_mean:.2f}/5.0\n"
                    
                    # Interpret the mean
                    if rec_mean <= 1.5:
                        consensus = "STRONG BUY"
                    elif rec_mean <= 2.5:
                        consensus = "BUY"
                    elif rec_mean <= 3.5:
                        consensus = "HOLD"
                    elif rec_mean <= 4.5:
                        consensus = "SELL"
                    else:
                        consensus = "STRONG SELL"
                    
                    result += f"- **Yahoo Finance Consensus**: {consensus}\n"
                
                # Get price targets
                if 'targetMeanPrice' in info:
                    result += f"- **Target Mean Price**: ${info['targetMeanPrice']:.2f}\n"
                if 'targetHighPrice' in info:
                    result += f"- **Target High Price**: ${info['targetHighPrice']:.2f}\n"
                if 'targetLowPrice' in info:
                    result += f"- **Target Low Price**: ${info['targetLowPrice']:.2f}\n"
                
                # Number of analysts
                if 'numberOfAnalystOpinions' in info:
                    result += f"- **Number of Analysts**: {info['numberOfAnalystOpinions']}\n"
            except Exception as e:
                result += f"Error retrieving Yahoo Finance data: {str(e)}\n"
        
        return result
    except Exception as e:
        return f"Error retrieving analyst recommendations for {ticker}: {str(e)}"