import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
from google import genai

from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.settings import GEMINI_API_KEY, logger
from daily_trading.utils.common import load_data

GEMINI_PRO = "gemini-3-pro-preview"
TEST_PROMPT = """
        Tell me about SMC trading OB, very short 1 sentence (5 words). Your response must be in the format:{"message": "your message here",
            "entry": null,
            "stop_loss": null,
            "take_profit": null
        }
        """


class PromptType(Enum):
    BOS_FVG = 1


@dataclass
class AssetData:
    symbol: str
    current_price: Optional[float] = None
    fvg_signals: Optional[list] = None
    bos_choch_signals: Optional[list] = None
    ob_signals: Optional[list] = None
    vol_profile: Optional[dict] = None
    high_lows: Optional[dict] = None
    ohlc_1d: Optional[dict] = None
    ohlc_4h: Optional[dict] = None
    ohlc_1h: Optional[dict] = None
    ohlc_15m: Optional[dict] = None
    ohlc_5m: Optional[dict] = None


@dataclass
class Ohlc:
    ohlc_1d: pd.DataFrame
    ohlc_4h: pd.DataFrame
    ohlc_1h: pd.DataFrame
    ohlc_15m: pd.DataFrame
    ohlc_5m: pd.DataFrame


class AITradeAssistant:
    def __init__(self, symbol: str, model=GEMINI_PRO, debug=False):
        self.api_key = GEMINI_API_KEY
        self.model = model
        self.symbol = symbol.upper()
        self.client = genai.Client()
        self.debug = debug

    @staticmethod
    async def build_bos_fvg_prompt(asset_data: AssetData) -> str:
        """Build the prompt for Gemini API based on ICT Smart Money Concepts."""
        prompt = (
            f"""
                    # You are the **SMC-Alpha Trading Agent**, a highly specialized algorithmic trader
                    executing a "Multi-Timeframe Trend Following" strategy based on Smart Money Concepts (SMC).
                    Your sole purpose is to analyze market data for {asset_data.symbol},
                    identify high-probability setups aligning Higher Timeframe (HTF) structure
                    with Lower Timeframe (LTF) entry signals, and output precise execution parameters.

        # EXECUTION AUTHORITY (NON-NEGOTIABLE)
        You are not a signal engine.
        You are a discretionary SMC trader operating inside a systematic framework.
        When rules conflict with market structure:
        Structure overrides precision
        Liquidity logic overrides optimization
        Trade skipping is a valid and expected outcome

        # STRATEGY CODEX: 1H-BOS-FVG
        ### 1. Higher Timeframe (HTF) Analysis [1-Hour Chart]
        *   **BULLISH BIAS:** Last confirmed signal was a Bullish BOS.
        *   **BEARISH BIAS:** Last confirmed signal was a Bearish BOS.
        *   **NEUTRAL:** Conflicting signals or range-bound.

        ### 2. Lower Timeframe (LTF) Execution [5-Minute Chart]
        *   **Trigger:** A "Fresh" Fair Value Gap (FVG) matching the HTF Bias.
        *   **Mitigation Logic (Smart Entry):**
            *   Wait for price to rebalance deep into the FVG to improve R:R.
            *   **MBT/NY:** Use 0.95 mitigation (deep pullbacks).
            *   **MNQ/NY:** Use 0.95 mitigation.
            *   **Others:** Use 0.75 default.

        ### 3. Structural Stop Loss Hierarchy (The "No-Man's-Land" Fix)
        To avoid arbitrary stops (math floors) and to avoid skipping too many trades (stops too wide), you must iterate through these tiers:

        **Tier 1: Candle Invalidation (Aggressive)**
        *   SL = Just beyond the "Candle 1" High/Low of the FVG.
        *   *Condition:* Is distance > 0.1%? If YES, use it. If NO (Too Tight), go to Tier 2.

        **Tier 2: Order Block Defense (The "Sweet Spot")**
        *   SL = Just beyond the Distal Line of the Order Block responsible for the FVG.
        *   *Condition:* Is `Size >= 1` with $250 risk? If YES, use it. If NO (Too Wide), go to Tier 3.

        **Tier 3: Swing Structure (Conservative)**
        *   SL = Just beyond the nearest Prior 5M Swing High/Low.
        *   *Condition:* Is `Size >= 1` with $250 risk? If YES, use it. If NO, **SKIP TRADE**.

        # INPUT DATA
        Here is the data you have access to for making your analysis:
        SYMBOL : {asset_data.symbol}
        1.  **HTF_Signal (BOS/CHoCH):** Latest 1H structure. \n {asset_data.bos_choch_signals} \n
        2.  **LTF_FVG_List:** 5M FVGs (Top, Bottom, Time, Mitigated). \n {asset_data.fvg_signals} \n
        3.  **5M Order Blocks:** (Crucial for Tier 2 SL). \n {asset_data.ob_signals} \n
        4.  **5M Highs/Lows:** (Crucial for Tier 3 SL). \n {asset_data.high_lows} \n
        5.  **Current_Price:** \n{asset_data.current_price}\n
        6.  **OHLC Data:** \n {asset_data.ohlc_5m} \n

        # DECISION PROCESS
        1. **Determine Bias**
           Establish the HTF directional bias using HTF_Signal BOS/CHoCH.
           Trades are only considered in the direction of this bias.

        2. **Scan for Opportunity**
           From LTF_FVG_List, identify Fair Value Gaps that:
           *   Align with the HTF bias
           *   Are unmitigated
           *   Are structurally valid relative to Current_Price (positioned where liquidity would logically be defended, not inside noise)

        3. **Construct the Trade**
           For the highest-quality FVG:
           *   **Entry:** Calculated using the defined mitigation tiers (e.g., 0.95 for MBT).
           *   **Stop Loss:** iterate through Tiers 1, 2, and 3. Select the first one that is **Structurally Safe** (not noise) AND **Financially Viable** (Size >= 1).
           *   **Take Profit:** Target 1.5R where structure allows. Minimum acceptable RR is 1.2R if structure limits expansion.

        ## Final Discretionary Validation
        Re-evaluate the trade as a human SMC execution:
        *   Reject stops that are mechanically tight (inside micro-range noise < 0.1%).
        *   Reject stops that are excessively wide (poor efficiency, Size < 1).
        *   Confirm the stop sits **beyond** liquidity (above a wick/high), not **inside** it.
        *   If structural protection is unclear or too expensive → **CANCEL/WAIT**.

        # OUTPUT FORMAT
        Respond with a JSON object ONLY.
        {{
          "decision": "ENTER_LONG" | "ENTER_SHORT" | "WAIT" | "CANCEL",
          "reasoning": "Explain the SL Tier selection (Wick vs OB vs Swing) and why the trade is valid.",
          "parameters": {{
            "symbol": "{{SYMBOL}}",
            "direction": "LONG" | "SHORT",
            "entry_price": 0.00,
            "stop_loss": 0.00,
            "take_profit": 0.00,
            "risk_reward_ratio": 1.2,
            "size": 1,
            "fvg_timestamp": "YYYY-MM-DD HH:MM:SS"
          }},
          "metadata": {{
            "sl_tier_used": "TIER_1_WICK" | "TIER_2_OB" | "TIER_3_SWING",
            "confidence_score": 0-100
          }}
        }}
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

    async def analyze_asset_trade_data(self, prompt_type: PromptType, asset_data: AssetData):
        if self.debug:
            return None, None
        if prompt_type == PromptType.BOS_FVG:
            my_prompt = await self.build_bos_fvg_prompt(asset_data)
        else:
            raise ValueError("Invalid prompt type specified.")
        response = await self.call_gemini_api(my_prompt)

        if isinstance(response, dict) and "error" in response:
            return "Failed to retrieve Gemini analysis", None

        if response is not None:
            try:
                logger.info(f"AI Usage: {response.usage_metadata.total_token_count} Total Tokens.")
                content = response.candidates[0].content.parts[0].text
                raw_message = json.loads(content.replace("```json\n", "").replace("\n```", "").strip())
                logger.info(f"Gemini AI analysis complete - {json.dumps(raw_message, indent=1)}")
                formatted_ai_message = self.format_ai_message(raw_message)
                return formatted_ai_message, raw_message
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing Gemini response: {e}")
                return "Error processing Gemini response", None
        return "Failed to retrieve Gemini analysis", None

    def format_ai_message(self, raw_message: dict) -> str:
        params = raw_message.get("parameters", {})
        metadata = raw_message.get("metadata", {})
        decision = raw_message.get("decision", "WAIT")

        symbol = params.get("symbol")
        if not symbol or symbol == "Unknown":
            symbol = self.symbol

        entry = params.get("entry_price")
        sl = params.get("stop_loss")
        tp = params.get("take_profit")
        size = params.get("size")
        rr = params.get("risk_reward_ratio")
        confidence = metadata.get("confidence_score")
        fvg_id = params.get("fvg_id") or metadata.get("fvg_id")
        bos_ts = params.get("bos_ts") or metadata.get("bos_ts")
        signal_ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        header = f"🚀 {decision.replace('_', ' ')} | {symbol}"
        tag = f"🏷️ 1H-BOS-FVG | {symbol}"

        return (
            f"{header}\n"
            f"{tag}\n"
            f"💲 Price: {entry}\n"
            f"🛡️ SL: {sl} | 🎯 TP: {tp}\n"
            f"📦 Size: {size}\n"
            f"RR: {rr} | Confidence: {confidence}%\n"
            f"⏱️ Signal (UTC): {signal_ts}\n"
            f"{{\n"
            f"  \"fvg_id\": {json.dumps(fvg_id)},\n"
            f"  \"bos_ts\": {json.dumps(bos_ts)}\n"
            f"}}"
        )


async def main():
    """
    Futures: GC, ES, NQ
    Crypto: BTCUSDT, ETHUSDT, XAUTUSDT
    """
    symbol = "MGC"
    want_to_pay_for_analyses = True  # change to False for free prompt only

    pathlib = "/project/daily_trading/bots/smc_bot/tests/ohlc_data/mbt_1h.csv"
    ohlc_1h = load_data(pathlib, start_ts="2025-10-01 00:00:00+00:00", end_ts="2025-10-25 00:00:00+00:00")
    pathlib = "/project/daily_trading/bots/smc_bot/tests/ohlc_data/mbt.csv"
    ohlc_5m = load_data(pathlib, start_ts="2025-10-01 00:00:00+00:00", end_ts="2025-10-25 00:00:00+00:00")
    # client = IBKRClient(host=HOST, port=PORT, client_id=CLIENT_ID)
    #
    # # 1. CONNECT
    # await client.connect()
    # await asyncio.sleep(2)
    # ohlc = await client.get_klines("MGC", "5m")

    fvg_detector = FvgDetector(main_tf="5m", pivot_period=5, fvg_threshold=0.25)
    fvg_results = fvg_detector.calculate_fvg(ohlc_5m)

    detector = BosChochDetector(main_tf="1h", show_bull=True, show_bear=True, left_bars=8, right_bars=3)
    results = detector.calculate_bos_choch(ohlc_1h)

    detector = SupplyDemandZonesDetector(fvg_based_only=False, swing_length=50, internal_length=5)
    active_swing, active_internal = detector.calculate(ohlc_5m)
    ob_results = active_swing + active_internal
    asset_data = AssetData(
        symbol=symbol,
        current_price=ohlc_5m["close"].iloc[-1],
        fvg_signals=fvg_results,
        ob_signals=ob_results,
        bos_choch_signals=results,
        ohlc_1h=ohlc_1h.tail(200).to_dict(orient="records"),
        ohlc_5m=ohlc_5m.tail(300).to_dict(orient="records"),
    )

    ai_agent = AITradeAssistant(symbol="MBT", model=GEMINI_PRO, debug=False)
    if want_to_pay_for_analyses:
        formatted_ai_message, raw_message = await ai_agent.analyze_asset_trade_data(prompt_type=PromptType.BOS_FVG, asset_data=asset_data)
        logger.info(formatted_ai_message)
    else:
        prompt = await ai_agent.build_bos_fvg_prompt(asset_data)
        logger.info(prompt)


if __name__ == "__main__":
    asyncio.run(main())
