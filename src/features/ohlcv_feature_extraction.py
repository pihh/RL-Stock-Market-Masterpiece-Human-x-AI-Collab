
def add_daily_features(df):
        df = df.sort_values("timestamp").reset_index(drop=True)
        df['date'] = df['timestamp'].dt.date

        df['day_of_month'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        # ABSOLUTE PARAMS
        df['candle_size']= df['high']-df['low']
        df['order_flow'] = (df['close'] - df['open']) 
        
        # CANDLESTICK COMPONENTS SIZE DISTRIBUTION
        df['candle_body'] = ((df['close'] - df['open']).abs()) / df['candle_size']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1))/ df['candle_size']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low'])/ df['candle_size']
        
        # ONE DAY VARIATION - to understand the direction and strenght
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['candle_change']= df['candle_size'].pct_change()
        df['vwap_change'] = df['vwap'].pct_change()
        df['trade_count_change'] = df['trade_count'].pct_change()
        df['order_flow_change'] = df['order_flow'].pct_change()
        df['overnight_price_change'] = df['open'] / df['close'].shift(1) - 1
        df['adjusted'] = df.isna().any(axis=1).astype(int)
        df = df.ffill()
        df = df.bfill()
        return df
"""
def ohlcv_cron():
    CSV_PATH = "./data/datasets/updated_sp500_ticker_history.csv"
    START_DATE = "2022-01-01"
    END_DATE = datetime.today()- timedelta(days=1)
    END_DATE = END_DATE.strftime("%Y-%m-%d")
    TICKERS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # --- Load ticker list ---
    sp500 = pd.read_html(TICKERS_URL)[0]
    tickers = sp500['Symbol'].tolist()
    tickers.append('SPY')
    
    #df = web.DataReader('^SPX', 'stooq', start='2022-01-01')
    #
    # --- Read existing CSV if exists ---
    ""
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH, parse_dates=["timestamp", "date"], low_memory=False)
        # Alpaca uses 'symbol' column for ticker
        print(f"Found existing CSV with {existing['symbol'].nunique()} tickers, {len(existing)} rows.")
    else:
        existing = pd.DataFrame()
        print("No existing CSV found; starting fresh.")
    ""
    # COMPUTE BASIC FEATURES
    def add_daily_features(df):
        df = df.sort_values("timestamp").reset_index(drop=True)
        df['date'] = df['timestamp'].dt.date

        df['day_of_month'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        # ABSOLUTE PARAMS
        df['candle_size']= df['high']-df['low']
        df['order_flow'] = (df['close'] - df['open']) 
        
        # CANDLESTICK COMPONENTS SIZE DISTRIBUTION
        df['candle_body'] = ((df['close'] - df['open']).abs()) / df['candle_size']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1))/ df['candle_size']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low'])/ df['candle_size']
        
        # ONE DAY VARIATION - to understand the direction and strenght
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['candle_change']= df['candle_size'].pct_change()
        df['vwap_change'] = df['vwap'].pct_change()
        df['trade_count_change'] = df['trade_count'].pct_change()
        df['order_flow_change'] = df['order_flow'].pct_change()
        df['overnight_price_change'] = df['open'] / df['close'].shift(1) - 1
        df['adjusted'] = df.isna().any(axis=1).astype(int)
        df = df.ffill()
        df = df.bfill()
        return df



    # --- Gather new data ---
    new_data = []
    for i, symbol in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] Processing {symbol} ...")
        # --- Lookback for previous day if updating ---
        prev_df = pd.DataFrame()
        existing_row = session.query(OHLCV).filter(OHLCV.symbol == symbol).order_by(desc(OHLCV.timestamp)).first()

        if existing_row:
            last_date = existing_row.timestamp  # This will be a string
            start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  Last date in DB: {last_date} | Downloading from: {start_date}")
            if pd.to_datetime(start_date) > pd.to_datetime(END_DATE):
                print("  Already up-to-date.")
                continue
            # Get the last row for this symbol, needed for correct pct_change etc.
            prev_df = fetch_ohlcv_to_df(session, symbol, start_date=last_date, end_date=last_date)
        else:
            start_date = START_DATE
            print(f"  No data found for {symbol}, starting at {START_DATE}")
            prev_df = pd.DataFrame()
        ""    
        if not existing.empty and symbol in existing['symbol'].values:
            last_date = existing.loc[existing['symbol']==symbol, 'timestamp'].max()
            start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  Last date in file: {last_date} | Downloading from: {start_date}")
            if pd.to_datetime(start_date) > pd.to_datetime(END_DATE):
                print("  Already up-to-date.")
                continue
            # Keep previous day's data to compute changes
            prev_rows = existing[(existing['symbol'] == symbol)].sort_values("timestamp").tail(1)
            if not prev_rows.empty:
                prev_df = prev_rows
        else:
            start_date = START_DATE
            print(f"  No data found for {symbol}, starting at {START_DATE}")
        ""
        try:
            if symbol.startswith("^"):
                bars= IndexAndIndicatorsRequest(
                    symbol,
                    start=start_date
                )
            else:
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=END_DATE
                )
                bars = client.get_stock_bars(request_params)
            if not bars.df.empty:
                df = bars.df.reset_index()
                df['symbol'] = symbol
                # Concatenate previous row (if present) for correct pct_change/overnight
                if not prev_df.empty:
                    df = pd.concat([prev_df, df], ignore_index=True)
                df = add_daily_features(df)
                # Drop first row (no price_change/overnight change possible)
                df = df.iloc[1:].copy()
                ew_df = pd.concat(new_data)
                upsert_ohlcv_from_df(session, new_df)
                new_data.append(df)
                print(f"  Downloaded {len(df)} new rows with features.")
            else:
                print("  No new data.")
        except Exception as e:
            print(f"  Failed to fetch {symbol}: {e}")
        time.sleep(0.35)

    # --- Append new data to CSV ---
    if new_data:
        new_df = pd.concat(new_data)
        #upsert_ohlcv_from_df(session, new_df)
        print("DB updated.")
    else:
        print("No new data.")

    # To fetch data for analysis:
    #aapl_df = fetch_ohlcv_to_df(session, 'AAPL', start_date='2024-01-01', end_date='2024-12-31')
    #print(aapl_df.head())

"""