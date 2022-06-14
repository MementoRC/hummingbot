# from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import pandas as pd


def Plot_OHCL(df):
    df_original = df.copy()
    # necessary convert to datetime
    df["Date"] = pd.to_datetime(df.Date)
    df["Date"] = df["Date"].apply(mpl_dates.date2num)

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # We are using the style ‘ggplot’
    plt.style.use('ggplot')

    # figsize attribute allows us to specify the width and height of a figure in unit inches
    fig = plt.figure(figsize=(16, 8))

    # Create top subplot for price axis
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

    # Create bottom subplot for volume which shares its x-axis
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

    # candlestick_ohlc(ax1, df.values, width=0.8 / 24, colorup='green', colordown='red', alpha=0.8)
    ax1.set_ylabel('Price', fontsize=12)
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    # Add Simple Moving Average
    ax1.plot(df["Date"], df_original['sma7'], '-')
    ax1.plot(df["Date"], df_original['sma25'], '-')
    ax1.plot(df["Date"], df_original['sma99'], '-')

    # Add Bollinger Bands
    ax1.plot(df["Date"], df_original['bb_bbm'], '-')
    ax1.plot(df["Date"], df_original['bb_bbh'], '-')
    ax1.plot(df["Date"], df_original['bb_bbl'], '-')

    # Add Parabolic Stop and Reverse
    ax1.plot(df["Date"], df_original['psar'], '.')

    # # Add Moving Average Convergence Divergence
    ax2.plot(df["Date"], df_original['MACD'], '-')

    # # Add Relative Strength Index
    ax2.plot(df["Date"], df_original['RSI'], '-')

    # beautify the x-labels (Our Date format)
    ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d'))  # %H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()

    plt.show()
