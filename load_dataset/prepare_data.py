import os
import pandas as pd

from Bybit_spot_futures import tokens, timeframes


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Расчёт ATR (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Расчёт RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Расчёт MACD (Moving Average Convergence Divergence)"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    return pd.DataFrame({'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist})


def process_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Обработка данных и расчёт индикаторов"""
    df['ma_20'] = df['close'].rolling(window=20).mean().round(2)
    df['ma_50'] = df['close'].rolling(window=50).mean().round(2)
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean().round(2)
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean().round(2)
    df['atr_14'] = calculate_atr(df, period=14).round(2)
    df['rsi_14'] = calculate_rsi(df['close'], period=14).round(2)
    # macd_df = calculate_macd(df['close'])
    # df['macd'] = macd_df['macd'].round(4)
    # df['macd_signal'] = macd_df['macd_signal'].round(4)
    # df['macd_hist'] = macd_df['macd_hist'].round(4)

    # result = df[["open_time", 'open', 'high', 'low', 'close', 'volume', 'ma_20', 'ma_50', 'ema_20', 'ema_50', 'atr_14', 'rsi_14', 'macd', 'macd_signal', 'macd_hist']].copy()
    result = df[[
        "open_time",
        "ticker",
        'open',
        'high',
        'low',
        'close',
        'volume',
        'ma_20',
        'ma_50',
        'ema_20',
        'ema_50',
        'atr_14',
        'rsi_14'
    ]].copy()
    # Убираем NaN значения от расчёта индикаторов
    return result.dropna()

if __name__ == '__main__':
    Data_path = "/home/dyadya/PycharmProjects/trade/nnPriceTrendPredictor/Data"

    for ticker in tokens:
        for timeframe_minutes in timeframes:
            tik = ticker.rstrip("USDT")
            os.makedirs(os.path.join(Data_path, f"{tik}_processed"), exist_ok=True)
            df = pd.read_csv(os.path.join(Data_path, f"_bybit_tkns_{timeframe_minutes}/{ticker}_{timeframe_minutes}.csv"))
            df['ticker'] = tik
            process_ohlcv(df).to_csv(os.path.join(Data_path, f"{tik}_processed/{ticker}_{timeframe_minutes}m.csv"), index=False)
