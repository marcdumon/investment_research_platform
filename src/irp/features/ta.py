import pandas as pd
import ta.momentum as _mom
import ta.trend as _trend
import ta.volatility as _vol
import ta.volume as _vlm


class TA:
    """Thin wrapper around the `ta` library for single-ticker price series."""

    def __init__(
        self,
        prices: pd.DataFrame,
        ticker: str | None = None,
        ticker_col: str = "ticker",
    ) -> None:
        df = prices[prices[ticker_col] == ticker].copy() if ticker else prices.copy()
        if df.empty:
            raise ValueError(f"No data for ticker {ticker!r}")
        df["date"] = pd.to_datetime(df["date"])
        self._df = df.sort_values("date").set_index("date")

    def _c(self) -> pd.Series:
        return self._df["close"]

    def _h(self) -> pd.Series:
        return self._df["high"]

    def _l(self) -> pd.Series:
        return self._df["low"]

    def _v(self) -> pd.Series:
        return self._df["volume"]

    # --- trend ---

    def sma(self, length: int = 50) -> pd.Series:
        s = _trend.SMAIndicator(self._c(), window=length).sma_indicator()
        s.name = f"SMA_{length}"
        return s

    def ema(self, length: int = 20) -> pd.Series:
        s = _trend.EMAIndicator(self._c(), window=length).ema_indicator()
        s.name = f"EMA_{length}"
        return s

    # --- momentum ---

    def rsi(self, length: int = 14) -> pd.Series:
        s = _mom.RSIIndicator(self._c(), window=length).rsi()
        s.name = f"RSI_{length}"
        return s

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        ind = _trend.MACD(self._c(), window_fast=fast, window_slow=slow, window_sign=signal)
        tag = f"{fast}_{slow}_{signal}"
        return pd.DataFrame({
            f"MACD_{tag}": ind.macd(),
            f"MACDh_{tag}": ind.macd_diff(),
            f"MACDs_{tag}": ind.macd_signal(),
        }, index=self._df.index)

    # --- volatility ---

    def bbands(self, length: int = 20, std: float = 2.0) -> pd.DataFrame:
        ind = _vol.BollingerBands(self._c(), window=length, window_dev=std)
        tag = f"{length}_{std}"
        return pd.DataFrame({
            f"BBL_{tag}": ind.bollinger_lband(),
            f"BBM_{tag}": ind.bollinger_mavg(),
            f"BBU_{tag}": ind.bollinger_hband(),
            f"BBB_{tag}": ind.bollinger_wband(),
            f"BBP_{tag}": ind.bollinger_pband(),
        }, index=self._df.index)

    def atr(self, length: int = 14) -> pd.Series:
        s = _vol.AverageTrueRange(self._h(), self._l(), self._c(), window=length).average_true_range()
        s.name = f"ATR_{length}"
        return s

    # --- volume ---

    def obv(self) -> pd.Series:
        s = _vlm.OnBalanceVolumeIndicator(self._c(), self._v()).on_balance_volume()
        s.name = "OBV"
        return s

    def vwap(self, length: int = 14) -> pd.Series:
        s = _vlm.VolumeWeightedAveragePrice(
            self._h(), self._l(), self._c(), self._v(), window=length
        ).volume_weighted_average_price()
        s.name = f"VWAP_{length}"
        return s

    # --- convenience ---

    def bundle(
        self,
        rsi: int = 14,
        sma: tuple[int, ...] = (20, 50, 200),
        ema: tuple[int, ...] = (20, 50),
        macd: bool = True,
        bbands: int = 20,
        atr: int = 14,
    ) -> pd.DataFrame:
        """Compute standard indicator set; return as wide DataFrame indexed by date."""
        parts: list[pd.Series | pd.DataFrame] = []
        for length in sma:
            parts.append(self.sma(length))
        for length in ema:
            parts.append(self.ema(length))
        if rsi:
            parts.append(self.rsi(rsi))
        if macd:
            parts.append(self.macd())
        if bbands:
            parts.append(self.bbands(bbands))
        if atr:
            parts.append(self.atr(atr))
        return pd.concat(parts, axis=1)
