import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd
import plotly.graph_objects as go
import pytz
from google import genai

from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.LevelsDetector import LevelsDetector
from daily_trading.bots.smc_bot.detectors.OrderBlocksDetectorOld import (
    OrderBlocksDetectorOld,
)
from daily_trading.bots.smc_bot.detectors.VolumeProfiler import VolumeProfiler
from daily_trading.settings import GEMINI_API_KEY, logger
from daily_trading.technicals.CryptoSentimentAnalyzer import CryptoSentimentAnalyzer

# GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_PRO = "gemini-3-pro-preview"  # "gemini-2.5-pro"
TEST_PROMPT = """
        Tell me about SMC trading OB, very short 1 sentence (5 words). Your response must be in the format:{"message": "your message here",
            "entry": null,
            "stop_loss": null,
            "take_profit": null
        }
        """


class PromptType(Enum):
    ICT_SMC = 1
    BREAK_OR_BOUNCE = 2
    TRENDLINE = 3


@dataclass
class AssetData:
    symbol: str
    sentiment: dict
    fvg_signals: list
    bos_choch_signals: list
    pivot_high_points: list
    pivot_low_points: list
    ob_signals: list
    vol_profile: dict
    support_resistance_data: dict
    ohlc_1d: dict
    ohlc_1h: dict
    ohlc_4h: dict
    ohlc_15m: dict


@dataclass
class Ohlc:
    ohlc_1d: pd.DataFrame
    ohlc_4h: pd.DataFrame
    ohlc_1h: pd.DataFrame
    ohlc_15m: pd.DataFrame


class GeminiTradeAssistant:
    def __init__(self, symbol: str, model: str, ohlc: Ohlc, main_tf="1h", debug=False):
        self.api_key = GEMINI_API_KEY
        self.model = model
        self.symbol = symbol.upper()
        self.ohlc_1h = ohlc.ohlc_1h
        self.ohlc_15m = ohlc.ohlc_15m
        self.ohlc_4h = ohlc.ohlc_4h
        self.ohlc_1d = ohlc.ohlc_1d
        self.main_tf = main_tf
        self.client = genai.Client()
        self.debug = debug

    async def fetch_data(self) -> AssetData:
        """Fetch OHLC mgc and calculate ICT signals."""

        symbol = self.ohlc_1d["symbol"].iloc[0]

        # Select the last 24 candles for 1H and 96 candles for 15m
        ohlc_1d_subset = self.ohlc_1h.tail(10).to_dict(orient="records")
        ohlc_1h_subset = self.ohlc_1h.tail(24).to_dict(orient="records")
        ohlc_15m_subset = self.ohlc_15m.tail(96).to_dict(orient="records")
        ohlc_4h_subset = self.ohlc_4h.tail(18).to_dict(orient="records")

        detector_fvg = FvgDetector(main_tf="15min", pivot_period=5, left_bars=5)
        detector_bos = BosChochDetector(self.main_tf, show_bull=True, show_bear=True)
        high_low_detector = HighLowDetector(self.main_tf)
        ob_detector = OrderBlocksDetectorOld(self.main_tf)
        support_levels_detector = LevelsDetector(main_tf="15m", pivot_length=3, strength=4)
        vol_profiler = VolumeProfiler(main_tf="1h")
        analyzer = CryptoSentimentAnalyzer()
        sentiment_summary = analyzer.format_for_trading_prompt(symbol)

        logger.info(f"Sentiment: {sentiment_summary}")

        bos_results = detector_bos.calculate_bos_choch(self.ohlc_1h, plot=False)
        fvg_results = detector_fvg.calculate_fvg(self.ohlc_15m, plot=False)
        ob_results = ob_detector.calculate_order_blocks(self.ohlc_1h, plot=False)
        vol_profiler_results = vol_profiler.calculate_volume_profile(self.ohlc_4h, plot=False)
        support_resistance_results = support_levels_detector.calculate_support_resistance_levels(self.ohlc_4h, plot=False)

        del vol_profiler_results["fig"]

        return AssetData(
            symbol=symbol,
            sentiment=sentiment_summary,
            fvg_signals=fvg_results["signals"],
            bos_choch_signals=bos_results["signals"],
            pivot_high_points=high_low_detector.calculate_pivots(self.ohlc_1h)[0],
            pivot_low_points=high_low_detector.calculate_pivots(self.ohlc_1h)[1],
            ob_signals=ob_results["order_blocks"],
            ohlc_1d=ohlc_1d_subset,
            ohlc_1h=ohlc_1h_subset,
            ohlc_15m=ohlc_15m_subset,
            ohlc_4h=ohlc_4h_subset,
            vol_profile=vol_profiler_results,
            support_resistance_data=support_resistance_results,
        )

        #     async def build_prompt(self) -> str:
        #         """Build the prompt for Gemini API based on ICT Smart Money Concepts."""
        #         mgc = await self.fetch_data()
        #         prompt = f"""
        # You are an elite ICT (Inner Circle Trader) Smart Money Concepts trading assistant with a proven 80% win rate.
        # Your expertise lies in analyzing institutional order flow, market structure, volume profile analysis, and market sentiment integration to identify high-probability trading opportunities.
        #
        # ## Your Trading Methodology
        # You follow strict ICT Smart Money Concepts combined with advanced market analysis:
        # - Market Structure Analysis (BOS/ChoCH) across multiple timeframes
        # - Fair Value Gap (FVG) identification
        # - Liquidity sweeps and stop hunts at pivot levels
        # - Order block analysis with institutional confirmation
        # - Premium/Discount array mapping
        # - Volume Profile analysis for institutional activity zones
        # - Market sentiment integration for contrarian opportunities
        # - Multi-timeframe confluence for precision entries
        #
        # ## Input Data:
        # - **Crypto Coin**: {mgc.symbol}
        # - **OHLC 1D Data**: {mgc.ohlc_1d}
        # - **OHLC 4H Data**: {mgc.ohlc_4h}
        # - **OHLC 1H Data**: {mgc.ohlc_1h}
        # - **OHLC 15MIN Data** - Entry timeframe for precise entry, SL, TP execution: {mgc.ohlc_15m}
        # - **15MIN FVG Signals** - Fair Value Gap locations for precise entry, SL, TP calculation: {mgc.fvg_signals}
        # - **1H BOS (Break of Structure) Signals**: {mgc.bos_choch_signals}
        # - **1H Pivot Highs** - Recent swing highs: {mgc.pivot_high_points}
        # - **1H Pivot Lows** - Recent swing lows: {mgc.pivot_low_points}
        # - **1H Order Blocks** - Institutional order blocks: {mgc.ob_signals}
        # - **4H Volume Profile** - Volume Profile insights into trading activity using 4H OHLC: {mgc.vol_profile}
        # - **Market Sentiment Analysis**: {mgc.sentiment}""" + """
        #
        # ## Analysis Framework:
        #
        # ### 1. Higher Timeframe Structure Analysis (4H/1D)
        # - **Daily Trend Direction**: Overall market bias and major institutional levels
        # - **1H Market Structure**: Primary trend, key swing points, and structural breaks
        # - **Higher Timeframe Bias**: [Bullish/Bearish/Neutral] based on multi-timeframe confluence
        # - **Structural Confirmation**: How 4H structure aligns with 1H BOS/ChoCH signals
        #
        # ### 2. Volume Profile & Institutional Activity
        # - **Point of Control (POC)**: Highest volume node - critical institutional level
        # - **Value Area High/Low (VAH/VAL)**: Key institutional acceptance/rejection zones
        # - **Volume Distribution**: Accumulation vs. distribution patterns
        # - **High/Low Volume Nodes**: Breakout zones and institutional interest areas
        # - **Smart Money Footprint**: How volume profile reveals institutional positioning
        #
        # ### 3. Market Sentiment Integration
        # - **Fear & Greed Index**: Market psychology for contrarian opportunities
        # - **News Sentiment**: Recent developments affecting {mgc.symbol}
        # - **Composite Sentiment**: Overall sentiment score and institutional divergence
        # - **Sentiment-Structure Confluence**: When sentiment aligns with or contradicts technical structure
        # - **Risk Assessment**: How sentiment affects position sizing and institutional behavior
        #
        # ### 4. ICT Smart Money Concepts Analysis
        # - **Market Structure**: Current BOS/ChoCH state and institutional flow direction
        # - **Order Blocks**: Most relevant institutional accumulation/distribution zones
        # - **Fair Value Gaps**: Active FVGs requiring mitigation and their directional bias
        # - **Liquidity Zones**: Pivot highs/lows where retail stops are clustered
        # - **Premium/Discount**: Current price position relative to institutional value areas
        # - **Institutional Candle Patterns**: Rejection wicks, engulfing patterns at key levels
        #
        # ### 5. Multi-Timeframe Confluence Assessment
        # - **1H Structure**: Primary institutional bias and key structural levels
        # - **15MIN Execution**: Precise entry timing using lower timeframe institutional signals
        # - **Timeframe Alignment**: How all timeframes support institutional flow direction
        # - **Confluence Zones**: Areas where multiple institutional concepts converge
        #
        # ## Trading Decision Framework:
        #
        # ### Entry Criteria - ALL Must Be Present:
        # - **Clear Higher Timeframe Bias**: 1H structure showing institutional direction
        # - **Multiple Confluence**: FVG + Order Block + BOS/ChoCH + Volume Profile alignment
        # - **Clean Market Structure**: Defined institutional levels with clear invalidation
        # - **Risk:Reward Minimum**: 1:2 ratio with structure-based targets
        # - **No Conflicting Signals**: All analysis components align with trade direction
        # - **Sentiment Confirmation**: Market psychology supports or provides contrarian opportunity
        #
        # ### Trade Setup Analysis:
        # - **Entry Point**: Exact price level based on institutional confluence
        # - **Stop Loss Placement**: Logical invalidation beyond institutional structure
        # - **Take Profit Targets**: Based on FVG levels, order blocks, and volume profile nodes
        # - **Entry Trigger**: Specific institutional signal confirming the setup
        #
        # ### Risk Management & Trade Execution:
        # - **Stop Loss Logic**: Why this level invalidates institutional thesis
        # - **Profit Target Reasoning**: Technical basis using institutional concepts
        # - **Trade Management**: Partial profits at institutional levels
        # - **Session Timing**: Optimal execution based on volume and volatility
        #
        # ## Response must short, clear, precise and easy to understand. Use abbreviations (i.e FVG) instead full names..
        # ## Response Output Format (JSON object):
        #
        # ### For HIGH-PROBABILITY SETUPS ONLY:
        # ```json
        # {
        #     "message": "• **Setup Summary**: [Clear trade direction and institutional reasoning]
        # • **Key Confluence**: [2-3 main institutional factors supporting the trade]
        # • **Entry Logic**: [Why this exact price level with ICT concepts]
        # • **Volume Profile Context**: [How POC/VAH/VAL supports the setup]
        # • **Sentiment Edge**: [How market psychology adds to or contradicts setup]
        # • **Risk Management**: [SL reasoning and R:R ratio]
        # • **Expected Move**: [TP target based on institutional levels]
        # • **Trade Quality**: [High/Medium - explain institutional strength]",
        #     "entry": [exact_price_integer],
        #     "take_profit": [exact_price_integer],
        #     "stop_loss": [exact_price_integer],
        # }
        # ```
        #
        # ### For NO CLEAR SETUP:
        # ```json
        # {
        #     "message": "• **No Clear Entry Opportunity**
        # • **Conflicting Signals**: [Explain specific conflicts in analysis]
        # • **What's Missing**: [Which confluence factors are absent]
        # • **Market State**: [Choppy/ranging/transitional structure issues]
        # • **Waiting For**: [Specific conditions needed for high-probability setup]
        # • **Sentiment Context**: [How current psychology affects waiting strategy]
        # • **Next Watch Levels**: [Key institutional levels to monitor]",
        #     "entry": null,
        #     "take_profit": null,
        #     "stop_loss": null,
        #
        # }
        # ```
        #
        # ## Trading Rules You Follow:
        # - **Quality Over Quantity**: Only trade with clear institutional bias and multiple confluence
        # - **Structure First**: Never trade against higher timeframe institutional flow
        # - **Risk Management**: Never risk more than 1-2% per trade, adjusted for sentiment
        # - **Patience Discipline**: Wait for perfect institutional setups rather than force trades
        # - **Volume Profile Integration**: Use institutional activity zones for entries and exits
        # - **Sentiment Awareness**: Factor market psychology into institutional analysis
        # - **Invalidation Clarity**: Always define what breaks your institutional thesis
        # - TP must be realistic based market structure and institutional levels
        #
        # ## Your 80% Win Rate Philosophy:
        # "My success comes from combining ICT Smart Money Concepts with volume profile analysis and sentiment integration.
        # I only take trades where institutional order flow is clear, multiple timeframes align, volume shows institutional interest, and sentiment either supports or provides contrarian opportunity.
        # When institutional signals conflict or structure is unclear, I wait.
        # Quality institutional setups with proper confluence consistently outperform quantity."
        #
        # Remember: Your edge comes from identifying where institutions are positioned and how retail
        # traders will be trapped. Use volume profile to confirm institutional activity,
        # sentiment analysis to gauge retail positioning, and ICT concepts to time entries with institutional flow.
        # When in doubt, preserve capital and wait for the next high-probability institutional setup.
        # """
        #         return prompt

    async def build_prompt(self) -> str:
        """Build the prompt for Gemini API based on ICT Smart Money Concepts."""
        data = await self.fetch_data()
        prompt = (
            f"""
You are an elite ICT (Inner Circle Trader) Sniper.
Your goal is to identify **Precision Limit Orders** at specific Institutional Arrays (FVGs, Order Blocks, Breakers).
You prioritize **High Win Rate** and **Realistic Targets** ("Low Hanging Fruit") over massive R:R ratios.

## Input Data:
- **Crypto Coin**: {data.symbol}
- **OHLC 1D**: {data.ohlc_1d}
- **OHLC 4H**: {data.ohlc_4h}
- **OHLC 1H**: {data.ohlc_1h}
- **OHLC 15M (Execution)**: {data.ohlc_15m}
- **15M FVG Signals**: {data.fvg_signals}
- **1H Structure**: {data.bos_choch_signals}
- **Pivot Highs/Lows**: {data.pivot_high_points} / {data.pivot_low_points}
- **Order Blocks**: {data.ob_signals}
- **Volume Profile**: {data.vol_profile}
- **Sentiment**: {data.sentiment}"""
            + """

## Strategic Instructions (The "Sniper" Mindset):
1.  **Limit Orders are King**: If price is currently expanding (chasing), DO NOT say "No Entry". Instead, identify the **Limit Order** waiting at the Premium/Discount array (FVG/OB) that caused the move. Set the entry there.
2.  **Tight Invalidation (SL)**: Do not use wide 4H swing stops. Place the Stop Loss just beyond the **15M Candle** or **Order Block** that provides the entry. If that level breaks, the specific setup is void.
3.  **Realistic Targets (TP)**: Target the **NEAREST** opposing liquidity pool or FVG (Standard 20-40 ticks). Do not aim for full trend reversals. Bank the profit.
4.  **Trend Alignment**: Only trade in the direction of the 1H/4H displacement.

## Analysis Checklist:
1.  **Determine Bias**: Look at 1H/4H Structure. (Bearish = Sell Rallies, Bullish = Buy Dips).
2.  **Identify Array**: Find the nearest unmitigated **15M FVG** or **Order Block** in the direction of the trend.
3.  **Volume Confirmation**: Does the Volume Profile support this level (e.g., edges of Value Area)?
4.  **Construct Setup**: 
    *   **Entry**: Limit at the open/50% of the FVG/OB.
    *   **SL**: Aggressive. Just outside the specific array.
    *   **TP**: The most recent local swing low/high (Internal Liquidity).

## Response Format (JSON):

### For A VALID SETUP (Limit or Market):
```json
{
    "message": "• **Setup Summary**: [Bearish/Bullish Limit Order at X]
    \n• **Key Confluence**: [Structure + Specific Array]
    \n• **Entry Logic**: [Why this specific FVG/OB is the reloading point]
    \n• **Volume Profile**: [Confluence with POC/VAH/VAL]\n• **Sentiment**: [Context]
    \n• **Risk Management**: [Strict SL logic based on candle invalidation]
    \n• **Expected Move**: [Nearest liquidity target]\n• **Trade Quality**: [High/Medium]",
    "entry": [exact_price_integer],
    "take_profit": [exact_price_integer],
    "stop_loss": [exact_price_integer]
}
For NO SETUP (Only if NO arrays exist):
code
JSON
{
    "message": "• **No Clear Entry Opportunity**\n• **Reason**: [No valid FVG or Order Block identified]\n• **Market State**: [e.g., Overextended with no resting liquidity]\n• **Next Watch**: [Level]",
    "entry": null,
    "take_profit": null,
    "stop_loss": null
}
Rules:
Output JSON Only.
SL must be realistic: Tight enough to allow high leverage, loose enough to survive a wick. (Typically 5-15 points on GC).
If current price is far from Entry: Output the SETUP anyway as a Limit Order. Do not say "Waiting". Give the numbers.
"""
        )

        return prompt

    async def call_gemini_api(self, prompt: str):
        try:
            logger.warning("$$$$$ Making Gemini API Request $$$$")
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            return response
        except Exception as e:
            logger.error(f"Gemini API failed: {e}")
            return {"error": f"API call failed: {e}"}

    async def analyze_asset_trade_data(self, prompt_type=PromptType.ICT_SMC):
        if prompt_type == PromptType.ICT_SMC:
            my_prompt = await self.build_prompt()
        else:
            raise ValueError("Invalid prompt type specified.")
        if self.debug:
            my_prompt = TEST_PROMPT
        response = await self.call_gemini_api(my_prompt)

        if "error" not in response:
            try:
                logger.info(f"AI Usage: {response.usage_metadata.total_token_count} Total Tokens.")
                content = response.candidates[0].content.parts[0].text
                raw_message = json.loads(content.replace("```json\n", "").replace("\n```", "").strip())
                logger.info(f"Gemini AI analysis complete - {json.dumps(raw_message, indent=1)}")
                formatted_ai_message = (
                    f"\n*** 🤖📊 AI Analysis | {self.symbol} ***\n"
                    f"\n{raw_message['message']}\n"
                    f"\nEntry: {raw_message['entry']}"
                    f"\nTP: {raw_message['take_profit']}"
                    f"\nSL: {raw_message['stop_loss']}"
                )
                return formatted_ai_message, raw_message
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing Gemini response: {e}")
                return {
                    "message": "Error processing Gemini response",
                    "entry": None,
                    "stop_loss": None,
                    "take_profit": None,
                }
        return {
            "message": "Failed to retrieve Gemini analysis",
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
        }

    @staticmethod
    def update_plot_with_entry_data(fig: go.Figure, raw_message):
        entry = raw_message["entry"]
        stop_loss = raw_message["stop_loss"]
        take_profit = raw_message["take_profit"]
        # Get x-axis range from candlestick mgc
        x_range = fig.data[0]["x"]  # Assumes candlestick is first trace
        x_start, x_end = x_range[0], x_range[-1]

        # Calculate midpoint for text placement
        x_mid = x_range[len(x_range) // 2]

        # Add Entry line (Dashed Orange with centered text)
        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[entry, entry],
                mode="lines",
                name="Entry",
                line=dict(color="orange", width=2, dash="dash"),
            )
        )
        fig.add_annotation(
            x=x_mid,
            y=entry,
            text="Entry",
            showarrow=False,
            font=dict(size=18),
            xanchor="center",
            yanchor="middle",
        )

        # Add Stop Loss line (Dashed Red with centered text)
        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[stop_loss, stop_loss],
                mode="lines",
                name="Stop Loss",
                line=dict(color="red", width=2, dash="dash"),
            )
        )
        fig.add_annotation(
            x=x_mid,
            y=stop_loss,
            text="Stop Loss",
            showarrow=False,
            font=dict(size=18),
            xanchor="center",
            yanchor="middle",
        )

        # Add Take Profit line (Dashed Green with centered text)
        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[take_profit, take_profit],
                mode="lines",
                name="Take Profit",
                line=dict(color="green", width=2, dash="dash"),
            )
        )
        fig.add_annotation(
            x=x_mid,
            y=take_profit,
            text="Take Profit",
            showarrow=False,
            font=dict(size=18),
            xanchor="center",
            yanchor="middle",
        )
        return fig

    @staticmethod
    def within_ai_request_hours(start_hour=8, end_hour=22, end_minute=0):
        pacific = pytz.timezone("US/Pacific")
        now = datetime.now(pacific)
        current_time = now.hour * 60 + now.minute
        start_time = start_hour * 60
        end_time = end_hour * 60 + end_minute
        return start_time <= current_time <= end_time


async def main():
    from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase

    """
    Futures: GC, ES, NQ
    Crypto: BTCUSDT, ETHUSDT, XAUTUSDT
    """
    symbol = "MGC"
    want_to_pay_for_analyses = True  # change to False for free prompt only

    # ‼️⚠️‼️ CHANGE MARKET TYPE BASED ON SYMBOL ‼️⚠️‼️
    market_type = "futures"  # 🚨change to "futures" for futures symbols like GC, ES, NQ

    bot = SMCBotBase(symbols=[symbol], main_tf="1h", market_type=market_type)

    # Fetch OHLC mgc within async context

    ohlc_1d = await bot.fetch_bitunix_ohlc_data(symbol, "1d", num_candles=10)
    ohlc_1h = await bot.fetch_bitunix_ohlc_data(symbol, "1h", num_candles=120)
    ohlc_15m = await bot.fetch_bitunix_ohlc_data(symbol, "15m", num_candles=200)
    ohlc_4h = await bot.fetch_bitunix_ohlc_data(symbol, "4h", num_candles=18)
    ohlc = Ohlc(ohlc_1d, ohlc_4h, ohlc_1h, ohlc_15m)

    ai_agent = GeminiTradeAssistant(symbol=symbol, model=GEMINI_PRO, ohlc=ohlc, debug=False)

    if want_to_pay_for_analyses:
        formatted_ai_message, raw_message = await ai_agent.analyze_asset_trade_data(prompt_type=PromptType.ICT_SMC)
        logger.info(formatted_ai_message)
    else:
        prompt = await ai_agent.build_prompt()
        logger.info(prompt)


if __name__ == "__main__":
    asyncio.run(main())
