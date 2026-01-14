import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.detectors.DetectorsBase import DetectorsBase


class LevelsDetector(DetectorsBase):
    def __init__(self, main_tf="1d", pivot_length=5, strength=1):
        super().__init__(main_tf, pivot_length, pivot_length)
        self.pivot_length = pivot_length
        self.strength = strength
        self.atr_length = 20
        self.too_close_atr = 1.0 / 8.0
        self.max_sr_info_size = 10
        self.atr = None

    def calculate_support_resistance_levels(self, ohlc, plot=True):
        ohlc = self.validate_data(ohlc)
        atr = self.calculate_atr(ohlc, self.atr_length)

        pivot_high = pd.Series(np.nan, index=ohlc.index)
        pivot_low = pd.Series(np.nan, index=ohlc.index)
        sr_info_list = []

        for i in range(self.pivot_length, len(ohlc) - self.pivot_length):
            left_window = ohlc.iloc[max(0, i - self.pivot_length):i]
            right_window = ohlc.iloc[i + 1:i + self.pivot_length + 1]
            if (left_window['high'].empty or ohlc['high'].iloc[i] >= left_window['high'].max()) and \
                    (right_window['high'].empty or ohlc['high'].iloc[i] >= right_window['high'].max()):
                pivot_high.iloc[i] = ohlc['high'].iloc[i]
                if self._is_valid_sr(ohlc['high'].iloc[i], sr_info_list, atr.iloc[i]):
                    sr_info_list.insert(0, {
                        'start_time': ohlc.index[i],
                        'price': ohlc['high'].iloc[i],
                        'type': 'Resistance',
                        'strength': 1
                    })
                    if len(sr_info_list) > self.max_sr_info_size:
                        sr_info_list.pop()

            if (left_window['low'].empty or ohlc['low'].iloc[i] <= left_window['low'].min()) and \
                    (right_window['low'].empty or ohlc['low'].iloc[i] <= right_window['low'].min()):
                pivot_low.iloc[i] = ohlc['low'].iloc[i]
                if self._is_valid_sr(ohlc['low'].iloc[i], sr_info_list, atr.iloc[i]):
                    sr_info_list.insert(0, {
                        'start_time': ohlc.index[i],
                        'price': ohlc['low'].iloc[i],
                        'type': 'Support',
                        'strength': 1
                    })
                    if len(sr_info_list) > self.max_sr_info_size:
                        sr_info_list.pop()

        for i in range(1, len(ohlc)):
            for sr in sr_info_list:
                if sr['type'] == 'Resistance' and ohlc['high'].iloc[i] >= sr['price'] and ohlc['close'].iloc[i] <= sr[
                    'price']:
                    sr['strength'] += 1
                elif sr['type'] == 'Support' and ohlc['low'].iloc[i] <= sr['price'] and ohlc['close'].iloc[i] >= sr[
                    'price']:
                    sr['strength'] += 1

        fig = None
        if plot:
            fig = self._plot_levels(ohlc, sr_info_list)

        pivot_high_points = [{'time': idx, 'level': pivot_high.loc[idx]} for idx in
                             pivot_high[pivot_high.notna()].index]
        pivot_low_points = [{'time': idx, 'level': pivot_low.loc[idx]} for idx in pivot_low[pivot_low.notna()].index]

        return {
            "support_levels": [sr for sr in sr_info_list if sr['type'] == 'Support'],
            "resistance_levels": [sr for sr in sr_info_list if sr['type'] == 'Resistance'],
            "pivot_high_points": pivot_high_points,
            "pivot_low_points": pivot_low_points,
            "fig": fig
        }

    def plot_htf_levels_on_ltf(self, ohlc_ltf, signals_htf):
        ohlc_ltf = self.validate_data(ohlc_ltf)
        ohlc_ltf = ohlc_ltf.sort_index().tail(self.plot_tail)

        symbol = ohlc_ltf['symbol'].iloc[0] if 'symbol' in ohlc_ltf.columns else 'Unknown'
        fig = go.Figure()

        # Plot candlesticks
        fig.add_trace(go.Candlestick(
            x=ohlc_ltf.index,
            open=ohlc_ltf['open'],
            high=ohlc_ltf['high'],
            low=ohlc_ltf['low'],
            close=ohlc_ltf['close'],
            name="OHLC"
        ))

        # Plot levels within OHLC range
        min_price = ohlc_ltf['low'].min()
        max_price = ohlc_ltf['high'].max()
        sr_levels = signals_htf.get('support_levels', []) + signals_htf.get('resistance_levels', [])
        for sr in sr_levels:
            if sr['strength'] >= self.strength and min_price <= sr['price'] <= max_price:
                start_time = sr['start_time']
                if start_time not in ohlc_ltf.index:
                    start_time = ohlc_ltf.index[ohlc_ltf.index.get_indexer([start_time], method='nearest')[0]]
                # Plot solid blue line
                fig.add_trace(go.Scatter(
                    x=[start_time, ohlc_ltf.index[-1]],
                    y=[sr['price'], sr['price']],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name=sr['type'],
                    showlegend=False
                ))
                # Add price text annotation
                fig.add_annotation(
                    x=ohlc_ltf.index[-1],
                    y=sr['price'],
                    text=f"{sr['price']}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=10, color='blue'),
                    xshift=5
                )

        last_candle = ohlc_ltf.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')

        if len(ohlc_ltf) > 1:
            interval = ohlc_ltf.index[-1] - ohlc_ltf.index[-2]
        else:
            interval = pd.Timedelta(minutes=1)
        x_range_max = ohlc_ltf.index[-1] + interval * 0.5

        tf_str = self.get_time_frame_name(ohlc_ltf)
        fig.update_layout(
            title=f"{symbol} |Support & Resistance Levels | Last Candle: {last_candle} (PT) | {tf_str} TF",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            xaxis=dict(range=[ohlc_ltf.index[0], x_range_max]),
            yaxis=dict(autorange=True),
            margin=dict(l=50, r=50, t=100, b=100),
        )
        fig.update_xaxes(
            type='date',
            tickformat="%H:%M",
            tickmode='linear',
            dtick=7200000,  # 2-hour intervals
            tickangle=45
        )

        return fig

    def _is_valid_sr(self, price, sr_info_list, atr):
        for sr in sr_info_list:
            if abs(sr['price'] - price) < atr * self.too_close_atr:
                return False
        return True

    def _plot_levels(self, ohlc, sr_info_list):
        tf_str = self.get_time_frame_name(ohlc)
        ohlc = ohlc.sort_index().tail(self.plot_tail)
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ), row=1, col=1)

        for sr in sr_info_list:
            if sr['strength'] >= self.strength:
                start_time = sr['start_time']
                if start_time not in ohlc.index:
                    start_time = ohlc.index[ohlc.index.get_indexer([start_time], method='nearest')[0]]
                end_time = ohlc.index[-1]
                fig.add_trace(go.Scatter(
                    x=[start_time, end_time],
                    y=[sr['price'], sr['price']],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name=sr['type'],
                    showlegend=False
                ), row=1, col=1)
                # Add price text annotation
                fig.add_annotation(
                    x=end_time,
                    y=sr['price'],
                    text=f"{sr['price']}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=10, color='blue'),
                    xshift=5
                )

        last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        # Extend x-axis to ensure last candle is fully visible
        x_range_end = ohlc.index[-1] + pd.Timedelta(minutes=120)  # Increased buffer to 30 minutes
        fig.update_layout(
            title=f"{symbol} | Support and Resistance Levels | Last Candle: {last_candle} (PT) | TF: {tf_str}",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            xaxis=dict(range=[ohlc.index[0], x_range_end]),
            margin=dict(l=50, r=50, t=100, b=100),
        )
        fig.update_xaxes(
            type='date',
            tickformat="%H:%M",
            tickmode='linear',
            dtick=7200000,  # 2-hour intervals
            tickangle=45,
            row=1, col=1
        )
        return fig
    def calculate_atr(self, ohlc, period):
        high_low = ohlc['high'] - ohlc['low']
        high_close = (ohlc['high'] - ohlc['close'].shift(1)).abs()
        low_close = (ohlc['low'] - ohlc['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.atr = true_range.rolling(window=period).mean()
        return self.atr
