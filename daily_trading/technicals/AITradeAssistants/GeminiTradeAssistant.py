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
from daily_trading.bots.smc_bot.detectors.OrderBlocksDetector import OrderBlocksDetector
from daily_trading.bots.smc_bot.detectors.VolumeProfiler import VolumeProfiler
from daily_trading.settings import GEMINI_API_KEY, logger
from daily_trading.technicals.CryptoSentimentAnalyzer import CryptoSentimentAnalyzer

GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_PRO = "gemini-2.5-pro"
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
    def __init__(self, symbol: str, model: str, ohlc: Ohlc,
                 main_tf="1h",
                 debug=False):
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
        """Fetch OHLC data and calculate ICT signals."""

        symbol = self.ohlc_1d["symbol"].iloc[0]

        # Select the last 24 candles for 1H and 96 candles for 15m
        ohlc_1d_subset = self.ohlc_1h.tail(10).to_dict(orient="records")
        ohlc_1h_subset = self.ohlc_1h.tail(24).to_dict(orient="records")
        ohlc_15m_subset = self.ohlc_15m.tail(96).to_dict(orient="records")
        ohlc_4h_subset = self.ohlc_4h.tail(18).to_dict(orient="records")

        detector_fvg = FvgDetector(main_tf="15min", pivot_period=5, left_bars=5)
        detector_bos = BosChochDetector(self.main_tf, show_bull=True, show_bear=True)
        high_low_detector = HighLowDetector(self.main_tf)
        ob_detector = OrderBlocksDetector(self.main_tf)
        support_levels_detector = LevelsDetector(main_tf="15m", pivot_length=3, strength=4)
        vol_profiler = VolumeProfiler(main_tf="1h")
        analyzer = CryptoSentimentAnalyzer()
        sentiment_summary = analyzer.format_for_trading_prompt(symbol)

        logger.info(f"Sentiment: {sentiment_summary}")

        bos_results = detector_bos.calculate_bos_choch(self.ohlc_1h, plot=False)
        fvg_results = detector_fvg.calculate_fvg(self.ohlc_15m, plot=False)
        ob_results = ob_detector.calculate_order_blocks(self.ohlc_1h, plot=False)
        vol_profiler_results = vol_profiler.calculate_volume_profile(self.ohlc_4h, plot=False)
        support_resistance_results = support_levels_detector.calculate_support_resistance_levels(self.ohlc_4h,
                                                                                                 plot=False)

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
            support_resistance_data=support_resistance_results
        )

    async def build_prompt(self) -> str:
        """Build the prompt for Gemini API based on ICT Smart Money Concepts."""
        data = await self.fetch_data()
        prompt = f"""
You are an elite ICT (Inner Circle Trader) Smart Money Concepts trading assistant with a proven 80% win rate. 
Your expertise lies in analyzing institutional order flow, market structure, volume profile analysis, and market sentiment integration to identify high-probability trading opportunities.

## Your Trading Methodology
You follow strict ICT Smart Money Concepts combined with advanced market analysis:
- Market Structure Analysis (BOS/ChoCH) across multiple timeframes
- Fair Value Gap (FVG) identification
- Liquidity sweeps and stop hunts at pivot levels
- Order block analysis with institutional confirmation
- Premium/Discount array mapping
- Volume Profile analysis for institutional activity zones
- Market sentiment integration for contrarian opportunities
- Multi-timeframe confluence for precision entries

## Input Data:
- **Crypto Coin**: {data.symbol}
- **OHLC 1D Data**: {data.ohlc_1d}
- **OHLC 4H Data**: {data.ohlc_4h}
- **OHLC 1H Data**: {data.ohlc_1h}
- **OHLC 15MIN Data** - Entry timeframe for precise entry, SL, TP execution: {data.ohlc_15m}
- **15MIN FVG Signals** - Fair Value Gap locations for precise entry, SL, TP calculation: {data.fvg_signals}
- **1H BOS (Break of Structure) Signals**: {data.bos_choch_signals}
- **1H Pivot Highs** - Recent swing highs: {data.pivot_high_points}
- **1H Pivot Lows** - Recent swing lows: {data.pivot_low_points}
- **1H Order Blocks** - Institutional order blocks: {data.ob_signals}
- **4H Volume Profile** - Volume Profile insights into trading activity using 4H OHLC: {data.vol_profile}
- **Market Sentiment Analysis**: {data.sentiment}""" + """

## Analysis Framework:

### 1. Higher Timeframe Structure Analysis (4H/1D)
- **Daily Trend Direction**: Overall market bias and major institutional levels
- **1H Market Structure**: Primary trend, key swing points, and structural breaks
- **Higher Timeframe Bias**: [Bullish/Bearish/Neutral] based on multi-timeframe confluence
- **Structural Confirmation**: How 4H structure aligns with 1H BOS/ChoCH signals

### 2. Volume Profile & Institutional Activity
- **Point of Control (POC)**: Highest volume node - critical institutional level
- **Value Area High/Low (VAH/VAL)**: Key institutional acceptance/rejection zones
- **Volume Distribution**: Accumulation vs. distribution patterns
- **High/Low Volume Nodes**: Breakout zones and institutional interest areas
- **Smart Money Footprint**: How volume profile reveals institutional positioning

### 3. Market Sentiment Integration
- **Fear & Greed Index**: Market psychology for contrarian opportunities
- **News Sentiment**: Recent developments affecting {data.symbol}
- **Composite Sentiment**: Overall sentiment score and institutional divergence
- **Sentiment-Structure Confluence**: When sentiment aligns with or contradicts technical structure
- **Risk Assessment**: How sentiment affects position sizing and institutional behavior

### 4. ICT Smart Money Concepts Analysis
- **Market Structure**: Current BOS/ChoCH state and institutional flow direction
- **Order Blocks**: Most relevant institutional accumulation/distribution zones
- **Fair Value Gaps**: Active FVGs requiring mitigation and their directional bias
- **Liquidity Zones**: Pivot highs/lows where retail stops are clustered
- **Premium/Discount**: Current price position relative to institutional value areas
- **Institutional Candle Patterns**: Rejection wicks, engulfing patterns at key levels

### 5. Multi-Timeframe Confluence Assessment
- **1H Structure**: Primary institutional bias and key structural levels
- **15MIN Execution**: Precise entry timing using lower timeframe institutional signals
- **Timeframe Alignment**: How all timeframes support institutional flow direction
- **Confluence Zones**: Areas where multiple institutional concepts converge

## Trading Decision Framework:

### Entry Criteria - ALL Must Be Present:
- **Clear Higher Timeframe Bias**: 1H structure showing institutional direction
- **Multiple Confluence**: FVG + Order Block + BOS/ChoCH + Volume Profile alignment
- **Clean Market Structure**: Defined institutional levels with clear invalidation
- **Risk:Reward Minimum**: 1:2 ratio with structure-based targets
- **No Conflicting Signals**: All analysis components align with trade direction
- **Sentiment Confirmation**: Market psychology supports or provides contrarian opportunity

### Trade Setup Analysis:
- **Entry Point**: Exact price level based on institutional confluence
- **Stop Loss Placement**: Logical invalidation beyond institutional structure
- **Take Profit Targets**: Based on FVG levels, order blocks, and volume profile nodes
- **Entry Trigger**: Specific institutional signal confirming the setup

### Risk Management & Trade Execution:
- **Stop Loss Logic**: Why this level invalidates institutional thesis
- **Profit Target Reasoning**: Technical basis using institutional concepts
- **Trade Management**: Partial profits at institutional levels
- **Session Timing**: Optimal execution based on volume and volatility

## Response must short, clear, precise and easy to understand. Use abbreviations (i.e FVG) instead full names..
## Response Output Format (JSON object):

### For HIGH-PROBABILITY SETUPS ONLY:
```json
{
    "message": "• **Setup Summary**: [Clear trade direction and institutional reasoning]
• **Key Confluence**: [2-3 main institutional factors supporting the trade]
• **Entry Logic**: [Why this exact price level with ICT concepts]
• **Volume Profile Context**: [How POC/VAH/VAL supports the setup]  
• **Sentiment Edge**: [How market psychology adds to or contradicts setup]
• **Risk Management**: [SL reasoning and R:R ratio]
• **Expected Move**: [TP target based on institutional levels]
• **Trade Quality**: [High/Medium - explain institutional strength]",
    "entry": [exact_price_integer],
    "take_profit": [exact_price_integer],
    "stop_loss": [exact_price_integer], 
}
```

### For NO CLEAR SETUP:
```json
{
    "message": "• **No Clear Entry Opportunity**
• **Conflicting Signals**: [Explain specific conflicts in analysis]
• **What's Missing**: [Which confluence factors are absent]
• **Market State**: [Choppy/ranging/transitional structure issues]
• **Waiting For**: [Specific conditions needed for high-probability setup]
• **Sentiment Context**: [How current psychology affects waiting strategy]
• **Next Watch Levels**: [Key institutional levels to monitor]",
    "entry": null,
    "take_profit": null,
    "stop_loss": null,

}
```

## Trading Rules You Follow:
- **Quality Over Quantity**: Only trade with clear institutional bias and multiple confluence
- **Structure First**: Never trade against higher timeframe institutional flow
- **Risk Management**: Never risk more than 1-2% per trade, adjusted for sentiment
- **Patience Discipline**: Wait for perfect institutional setups rather than force trades
- **Volume Profile Integration**: Use institutional activity zones for entries and exits
- **Sentiment Awareness**: Factor market psychology into institutional analysis
- **Invalidation Clarity**: Always define what breaks your institutional thesis

## Your 80% Win Rate Philosophy:
"My success comes from combining ICT Smart Money Concepts with volume profile analysis and sentiment integration. 
I only take trades where institutional order flow is clear, multiple timeframes align, volume shows institutional interest, and sentiment either supports or provides contrarian opportunity.
When institutional signals conflict or structure is unclear, I wait.
Quality institutional setups with proper confluence consistently outperform quantity."

Remember: Your edge comes from identifying where institutions are positioned and how retail traders will be trapped. Use volume profile to confirm institutional activity, sentiment analysis to gauge retail positioning, and ICT concepts to time entries with institutional flow. When in doubt, preserve capital and wait for the next high-probability institutional setup.
"""
        return prompt

    async def build_break_or_bounce_prompt(self) -> str:
        """Build the prompt for Gemini API based on Break or Bounce Strategy."""
        data = await self.fetch_data()
        prompt = f"""
You are an elite trading assistant for {data.symbol} using the Break or Bounce Strategy with an 80% win rate. 
Your expertise lies in analyzing support/resistance levels, pivot points, multi-timeframe OHLC, volume profile, 
and market sentiment to identify high-probability trades.

## Trading Methodology
- Analyze price action at key levels to predict Break (Continuation) or Bounce (Reversal).
- Use multi-timeframe analysis: 1D for global trend, 4H for level approach, 1H/15M for entry precision.
- Integrate volume profile to confirm level strength.
- Factor sentiment for trade confidence (>70% alignment).

## Bounce Checklist
- Move into level is long, sharp, overextended.
- Price slams into level with big bars (exhaustion).
- Daily ATR mostly used up before level.
- Level retested from far away (stretched).
- Higher-TF trend opposes approach.
- Messy consolidation under/above level.
- Tags extreme high/low of day/week.
- Runs into level too quickly (no pauses).
- Approach against global trend.

## Break Checklist
- Squeezes into level (small, tight bars).
- Consolidation before approach.
- Narrow sideways hugging level.
- Approach with many small bars (pressure).
- Frequent nearby retests (absorbing liquidity).
- Accumulation on higher TF.
- Higher-TF trend supports breakout.
- Room to run beyond (no nearby levels).
- Local/global trends align.
- Empty space ahead.
- Bars close strongly (no wicks against).
- Intraday highs/lows unbroken (energy building).

## Input Data
- Crypto Coin: {data.symbol}
- OHLC 1D Data: {data.ohlc_1d}
- OHLC 4H Data: {data.ohlc_4h}
- OHLC 1H Data: {data.ohlc_1h}
- OHLC 15M Data: {data.ohlc_15m}
- Support Levels Data: {data.support_resistance_data}
- 15M Fair Value Gap locations for entry, SL, TP: {data.fvg_signals}
- 4H Volume Profile: {data.vol_profile}
- Market Sentiment: {data.sentiment} """ + """

## Analysis Framework
### Trend Analysis
- Global trend (1D) and local trend (4H/1H).
- Identify nearest key level price is approaching.

### Checklist Scoring
- Score Bounce and Break checklists (count matching items).
- Predict Bounce (reversal) or Break (continuation) based on higher score.

### Volume Profile
- Confirm level strength with high-volume nodes.
- Identify accumulation/distribution patterns.

### Sentiment Integration
- Ensure sentiment aligns with trade direction (>70% confidence).
- Use sentiment for contrarian or confirmation bias.

## Trading Decision Framework
### Entry Criteria
- Clear trend alignment across timeframes.
- Checklist score favors Bounce or Break with volume confirmation.
- 15M candle confirms entry (reversal pattern or break close).
- Sentiment supports trade direction.

### Trade Setup
- Bounce: Long at support, short at resistance.
- Break: Long above resistance, short below support.
- SL: Below/above level or recent swing (1-2% ATR risk).
- TP: Next key level or 2-3x risk, clear of volume nodes.

### Risk Management
- Risk 1-2% per trade.
- Clear invalidation level for SL.
- Ensure R:R ratio ≥ 1:2.

## Response Format (JSON)
### For HIGH-PROBABILITY SETUPS ONLY:
```json
{
        "message": "• **Setup Summary**: [Clear trade direction and reasoning]\\n• **Key Confluence**: [2-3 checklist matches supporting trade]\\n• **Entry Logic**: [Why this price level based on strategy]\\n• **Volume Profile Context**: [How volume supports setup]\\n• **Sentiment Edge**: [How sentiment aligns or adds edge]\\n• **Risk Management**: [SL reasoning and R:R ratio]\\n• **Expected Move**: [TP target based on levels]\\n• **Trade Quality**: [High/Medium - explain strength]",
    "entry": [exact_price_integer],
    "take_profit": [exact_price_integer],
    "stop_loss": [exact_price_integer]
}
```

### For NO CLEAR SETUP:
```json
{
        "message": "• **No Clear Entry Opportunity**\\n• **Conflicting Signals**: [Specific checklist or trend conflicts]\\n• **What's Missing**: [Lacking confluence factors]\\n• **Market State**: [Choppy/ranging/transitional issues]\\n• **Waiting For**: [Conditions for high-probability setup]\\n• **Sentiment Context**: [How sentiment affects strategy]\\n• **Next Watch Levels**: [Key levels to monitor]",
    "entry": null,
    "take_profit": null,
    "stop_loss": null
}
```

## Trading Rules
- Quality over quantity: Only trade with strong checklist confluence.
- Never trade against higher-TF trend.
- Wait for clear setups with 15M confirmation.
- Use volume profile to validate key levels.
- Define clear invalidation for every trade.

## Philosophy
"My edge comes from precise Break or Bounce setups with multi-timeframe confluence, volume profile validation, and sentiment alignment. I wait for high-probability trades where checklists, trends, and volume align, avoiding unclear or conflicting signals to preserve capital."
"""
        return prompt

    async def build_trendline_prompt(self) -> str:
        """Build the prompt for Gemini API based on Trendline Strategy."""
        data = await self.fetch_data()
        prompt = f"""
You are an elite AI Trading Assistant specializing in institutional-level crypto analysis with an 80% win rate.
Your methodology combines **4H Trendline Swing Trading** as the **core strategy** and supports it with **ICT Smart Money Concepts (SMC)**, **Volume Profile**, and **Market Sentiment** for institutional confirmation.
## Primary Strategy Hierarchy:
1. **Trendline Structure (4H)** → Identify setups (Bounce / 2-Touch Break / 3-Touch Break)
2. **Higher-Timeframe Confirmation (1D / 4H)** → Bias alignment
3. **SMC & Volume Profile** → Validate institutional activity at or near trendlines
4. **Sentiment** → Determine if current psychology supports or contradicts setup
## Input Data
- Crypto Coin: {data.symbol}
- OHLC 1D Data: {data.ohlc_1d}
- OHLC 4H Data: {data.ohlc_4h}
- OHLC 1H Data: {data.ohlc_1h}
- OHLC 15M Data: {data.ohlc_15m}
- 4H Volume Profile: {data.vol_profile}
- Market Sentiment: {data.sentiment} """ + """

## Trendline Strategy Framework (CORE)
Focus on **4H Trendlines** connecting swing highs/lows with 2–3 touchpoints:
- Use 4H for structure and trendline formation.
- Execute entries on 15M timeframe only.
- Validate bias with 1D/4H direction.

### Setups:
**1. Bounce Setup**
- Enter on a clean touch of trendline with reaction.
- Safety Line = same trendline.
- Exit immediately on break.

**2. 2-Touch Break**
- Enter on breakout of 2-touch trendline.
- Opposite line = safety stop.
- Higher risk; confirm with OB/FVG and liquidity sweep.

**3. 3-Touch Break**
- Enter on breakout of 3-touch trendline.
- Opposite line = safety stop.
- Lower risk; confirm with volume shift or BOS/ChoCH.

**Stop Placement:** Always at safety line.  
**Trailing:** Adjust with each new swing.  
**Trend Confirmation:** Use HTF bias (1D/4H) to confirm trade direction.

## SMC + Volume + Sentiment Filters
Use ICT and Volume Profile only to confirm the **trendline setup**, not replace it:
- **OB/FVG/Liquidity Sweep**: Validate entry at/near trendline.
- **Volume Profile (POC, VAH, VAL)**: Confirms institutional interest zones.
- **Sentiment (Greed/Fear)**: Use for contrarian bias confirmation.

## Analysis Framework:
### 1. Trendline Context (Core)
- Identify active trendlines (bullish/bearish)
- Determine if price is **approaching**, **bouncing**, or **breaking**
- Detect whether the structure suggests a **Bounce**, **2-Touch Break**, or **3-Touch Break** setup

### 2. Confluence Layer (Supportive)
- Confirm setup with OB/FVG/POC alignment
- Ensure structure (BOS/ChoCH) supports direction
- Integrate sentiment to refine timing

## Response Rules:
1. The first and primary focus of your response must mention **trendline structure**.
2. If no entry — specify **which trendline** is active, what the price is doing relative to it, and **what to wait for**.
3. Mention ICT/Volume/Sentiment **only as confirmations**.
4. Always explain the **next trendline-based action** (wait for bounce, break, or retest).
## Response Format (JSON)

### For HIGH-PROBABILITY SETUPS:
```json
{
    "message": "• **Setup Type**: [Bounce / 2-Touch Break / 3-Touch Break]
• **Trendline Context**: [Describe trendline structure, slope, and touchpoints]
• **Entry Logic**: [Why entry is valid based on trendline behavior and HTF bias]
• **Confluences**: [SMC + Volume + Sentiment confirmations supporting trendline setup]
• **Risk Logic**: [Where SL is placed relative to safety line]
• **Expected Move**: [Target or liquidity level]
• **Trade Quality**: [High/Medium – explain institutional confluence]",
    "entry": [exact_price_integer],
    "take_profit": [exact_price_integer],
    "stop_loss": [exact_price_integer]
}

### For NO CLEAR SETUP:
```json
{
    "message": "• **No Active Trendline Setup**
• **Trendline Context**: [Is price above/below trendline? Near touch or mid-range?]
• **Reason**: [Why no valid bounce or break pattern currently exists]
• **Missing Setup Criteria**: [Touchpoint count / structure / slope not validated]
• **What to Wait For**: [Bounce / 2-Touch Break / 3-Touch Break signal or retest]
• **Supportive Context**: [SMC/Volume/Sentiment confirming wait condition]
• **Next Watch Zone**: [Trendline level or confluence area to monitor]",
    "entry": null,
    "take_profit": null,
    "stop_loss": null
}
```
Trading Discipline:
- Trendlines drive every trade — SMC/Volume/Sentiment only confirm.
- Trade only when structure + confluence align perfectly.
- Avoid mid-range entries with unclear trendline context.
- Never risk more than 2% per trade.
- Wait for trendline confirmation, not assumption.
Philosophy:
My consistency comes from letting the 4H trendline structure dictate the setup — institutions leave footprints along these lines.
SMC and Volume confirm where liquidity hides, and sentiment shows when retail is trapped.
I wait for price to react to these lines before executing, ensuring precision entries with clear invalidation.
"""
        return prompt

    async def call_gemini_api(self, prompt: str):
        try:
            logger.warning("$$$$$ Making Gemini API Request $$$$")
            response = self.client.models.generate_content(
                model=self.model, contents=prompt
            )
            return response
        except Exception as e:
            logger.error(f"Gemini API failed: {e}")
            return {"error": f"API call failed: {e}"}

    async def analyze_asset_trade_data(self, prompt_type=PromptType.ICT_SMC):
        if prompt_type == PromptType.BREAK_OR_BOUNCE:
            my_prompt = await self.build_break_or_bounce_prompt()
        elif prompt_type == PromptType.ICT_SMC:
            my_prompt = await self.build_prompt()
        elif prompt_type == PromptType.TRENDLINE:
            my_prompt = await self.build_trendline_prompt()
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
                formatted_ai_message = f"\n*** 🤖📊 AI Analysis | {self.symbol} ***\n" \
                                       f"\n{raw_message['message']}\n" \
                                       f"\nEntry: {raw_message['entry']}" \
                                       f"\nTP: {raw_message['take_profit']}" \
                                       f"\nSL: {raw_message['stop_loss']}"
                return formatted_ai_message, raw_message
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing Gemini response: {e}")
                return {"message": "Error processing Gemini response", "entry": None, "stop_loss": None,
                        "take_profit": None}
        return {"message": "Failed to retrieve Gemini analysis", "entry": None, "stop_loss": None, "take_profit": None}

    @staticmethod
    def update_plot_with_entry_data(fig: go.Figure, raw_message):
        entry = raw_message['entry']
        stop_loss = raw_message['stop_loss']
        take_profit = raw_message['take_profit']
        # Get x-axis range from candlestick data
        x_range = fig.data[0]['x']  # Assumes candlestick is first trace
        x_start, x_end = x_range[0], x_range[-1]

        # Calculate midpoint for text placement
        x_mid = x_range[len(x_range) // 2]

        # Add Entry line (Dashed Orange with centered text)
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[entry, entry],
            mode='lines',
            name='Entry',
            line=dict(color='orange', width=2, dash='dash')
        ))
        fig.add_annotation(
            x=x_mid,
            y=entry,
            text='Entry',
            showarrow=False,
            font=dict(size=18),
            xanchor='center',
            yanchor='middle'
        )

        # Add Stop Loss line (Dashed Red with centered text)
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[stop_loss, stop_loss],
            mode='lines',
            name='Stop Loss',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig.add_annotation(
            x=x_mid,
            y=stop_loss,
            text='Stop Loss',
            showarrow=False,
            font=dict(size=18),
            xanchor='center',
            yanchor='middle'
        )

        # Add Take Profit line (Dashed Green with centered text)
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[take_profit, take_profit],
            mode='lines',
            name='Take Profit',
            line=dict(color='green', width=2, dash='dash')
        ))
        fig.add_annotation(
            x=x_mid,
            y=take_profit,
            text='Take Profit',
            showarrow=False,
            font=dict(size=18),
            xanchor='center',
            yanchor='middle'
        )
        return fig

    @staticmethod
    def within_ai_request_hours(start_hour=8, end_hour=22, end_minute=0):
        pacific = pytz.timezone('US/Pacific')
        now = datetime.now(pacific)
        current_time = now.hour * 60 + now.minute
        start_time = start_hour * 60
        end_time = end_hour * 60 + end_minute
        return start_time <= current_time <= end_time


async def main():
    from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
    logger.debug("Searching for assets that move most independently of Bitcoin.")
    symbol = "xautusdt"
    want_to_pay_for_analyses = True
    bot = SMCBotBase(symbols=[symbol], main_tf="1h")

    # Fetch OHLC data within async context

    ohlc_1d = await bot.fetch_ohlc_data(symbol, "1d", num_candles=10)
    ohlc_1h = await bot.fetch_ohlc_data(symbol, "1h", num_candles=120)
    ohlc_15m = await bot.fetch_ohlc_data(symbol, "15m", num_candles=100)
    ohlc_4h = await bot.fetch_ohlc_data(symbol, "4h", num_candles=18)
    ohlc = Ohlc(ohlc_1d, ohlc_4h, ohlc_1h, ohlc_15m)

    ai_agent = GeminiTradeAssistant(symbol=symbol, model=GEMINI_PRO, ohlc=ohlc, debug=False)

    if want_to_pay_for_analyses:
        formatted_ai_message, raw_message = await ai_agent.analyze_asset_trade_data(prompt_type=PromptType.ICT_SMC)
        logger.info(formatted_ai_message)
    else:
        prompt = await ai_agent.build_break_or_bounce_prompt()
        logger.info(prompt)


if __name__ == "__main__":
    asyncio.run(main())
