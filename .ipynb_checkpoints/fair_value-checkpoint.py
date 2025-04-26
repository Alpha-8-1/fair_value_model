import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import configparser
from fredapi import Fred

def load_config():
    config = configparser.ConfigParser()
    config.read('config/fred.cfg')
    return config['FRED']['api_key']

def pull_data(fred_api_key):
    fred = Fred(api_key=fred_api_key)

    forward_eps = 270  # manual input for now
    equity_risk_premium = 0.045

    sp500 = yf.Ticker("^GSPC")
    sp500_price = sp500.history(period="1d")['Close'].iloc[-1]

    tnx = yf.Ticker("^TNX")
    yield_10y = tnx.history(period="1d")['Close'].iloc[-1] / 100

    one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    cpi_series = fred.get_series('CPIAUCSL', observation_start=one_year_ago)
    cpi_latest = cpi_series.iloc[-1]
    cpi_last_year = cpi_series.iloc[0]
    cpi_yoy = ((cpi_latest - cpi_last_year) / cpi_last_year) * 100

    pmi = 49.8  # manual input
    vix = yf.Ticker("^VIX")
    vix_value = vix.history(period="1d")['Close'].iloc[-1]

    t10 = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1]
    t2 = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1]
    credit_spread_proxy = (t10 - t2) / 100

    sentiment_score = 0.6  # manual input

    return (forward_eps, equity_risk_premium, sp500_price, yield_10y, cpi_yoy, pmi, vix_value, credit_spread_proxy, sentiment_score)

def calculate_fair_value(forward_eps, equity_risk_premium, yield_10y, cpi_yoy, pmi, vix_value, credit_spread_proxy, sentiment_score):
    real_yield = yield_10y - (cpi_yoy / 100)
    fair_pe = 1 / (real_yield + equity_risk_premium)
    base_fair_value = forward_eps * fair_pe

    def inflation_penalty(cpi):
        return -1.5 * (cpi - 2) if cpi > 2 else 0

    def pmi_penalty(pmi):
        return -0.5 * (50 - pmi) if pmi < 50 else 0

    def vix_adjustment(vix):
        if vix > 20:
            return -5
        elif vix < 15:
            return 2
        else:
            return 0

    def credit_spread_adjustment(spread):
        extra = spread - 0.01
        return -(extra / 0.005) * 2 if extra > 0 else 0

    def sentiment_adjustment(sentiment_score):
        return 2 if sentiment_score > 0.5 else -2

    adjustments = (
        inflation_penalty(cpi_yoy) +
        pmi_penalty(pmi) +
        vix_adjustment(vix_value) +
        credit_spread_adjustment(credit_spread_proxy) +
        sentiment_adjustment(sentiment_score)
    )

    adjusted_fair_value = base_fair_value * (1 + adjustments / 100)

    return base_fair_value, adjusted_fair_value, adjustments

def plot_fair_value(sp500_price, adjusted_fair_value):
    fig, ax = plt.subplots()
    ax.bar(['Current S&P 500', 'Adjusted Fair Value'], [sp500_price, adjusted_fair_value], color=['blue', 'green'])
    ax.set_ylabel('Index Level')
    ax.set_title('S&P 500 Actual vs Model Fair Value')
    plt.grid(True)
    plt.show()

def main():
    fred_api_key = load_config()
    data = pull_data(fred_api_key)
    base_fair_value, adjusted_fair_value, adjustments = calculate_fair_value(*data[:-1])
    sp500_price = data[2]

    print(f"--- Fair Value Model Results as of {datetime.now().strftime('%Y-%m-%d')} ---")
    print(f"Current S&P 500: {sp500_price:.2f}")
    print(f"Base Model Fair Value: {base_fair_value:.2f}")
    print(f"Adjusted Fair Value: {adjusted_fair_value:.2f}")
    print(f"Total Adjustment Applied: {adjustments:.2f}%")
    print(f"Market is {'overvalued' if adjusted_fair_value < sp500_price else 'undervalued'} by {abs(sp500_price - adjusted_fair_value) / sp500_price * 100:.2f}%")

    plot_fair_value(sp500_price, adjusted_fair_value)

if __name__ == "__main__":
    main()
