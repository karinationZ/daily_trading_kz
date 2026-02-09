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
            Your sole purpose is to analyze market data, identify high-probability setups aligning Higher Timeframe (HTF) structure 
            with Lower Timeframe (LTF) entry signals, and output precise execution parameters. You do not gamble; you execute logic.

# STRATEGY CODEX: 1H-BOS-FVG
### 1. Higher Timeframe (HTF) Analysis [1-Hour Chart]
*   **Objective:** Determine the Market Bias.
*   **Logic:**
    *   **BULLISH BIAS:** The last confirmed signal was a Bullish Break of Structure (BOS).
    *   **BEARISH BIAS:** The last confirmed signal was a Bearish Break of Structure (BOS).
    *   **NEUTRAL:** No clear structure or conflicting signals.
*   **Constraint:** YOU MUST trade in the direction of the HTF Bias.

### 2. Lower Timeframe (LTF) Execution [5-Minute Chart]
*   **Objective:** Identify the Entry Trigger.
*   **Signal:** A "Fresh" Fair Value Gap (FVG) that matches the HTF Bias.
*   **Validation Rules (CRITICAL):**
    *   **Freshness:** The FVG must NOT be mitigated (price has not yet returned to fill it).
    *   **Recency:** Prioritize the most recent unmitigated FVG.
    *   **Zombie Check:** If the price is already far beyond the FVG (invalidated), ignore it.

### 3. Execution Parameters (Data-Driven Logic)
    You must calculate parameters based on backtest leaderboards for ALL instruments and sessions.
    Use the instrument+session-specific mitigation preference below as the default. If no match, use 0.75.
    Preferred Mitigation by Instrument+Session:
    - MNQ NY: 0.95 (fallback 0.75)
    - MNQ LONDON: 0.75
    - MNQ ASIA: 0.75
    - MES NY: 0.95
    - MES LONDON: 0.95
    - MES ASIA: 0.75
    - MGC NY: 0.95
    - MGC LONDON: 0.25
    - MGC ASIA: 0.50
    - MBT NY: 0.95
    - MBT LONDON: 0.95
    - MBT ASIA: 0.95
    A. Smart Entry (Deep Mitigation)
    Constraint: You must wait for price to rebalance 25% to 95% of the FVG (use the preferred mitigation above). Do not front-run the gap.
    Formulas:
    LONG: FVG_Top - (FVG_Height * Mitigation_Ratio)
    SHORT: FVG_Bottom + (FVG_Height * Mitigation_Ratio)
    B. Structural Stop Loss (SL)
    Placement: Structural Invalidation (Beyond Candle 1 High/Low).
    Buffer: Add 2.0 Points (MNQ/ES) to survive standard stop hunts.
    Minimum SL Distance:
    The SL MUST BE AT LEAST 0.1% of entry price away from entry.
    If |Entry - SL| / Entry < 0.001, widen SL to exactly 0.1% from entry.
    Soft Penalty (Not a Rejection):
    If |Entry - SL| / Entry > 0.002 (0.2%), you may still trade, but reduce confidence by 20 points
    and clearly mention "wide structural SL" in reasoning.
    Risk Logic (Static):
    Risk Per Trade: $250 USD (Static).
    Validation: Calculate the number of contracts allowed with $250 risk.
    Contracts = $250 / ( |Entry - SL| * Point_Value )
    If Contracts < 1 (Stop is too wide for $250 risk), REJECT the trade.
    C. Dynamic Take Profit (TP)
    Target RR: 1.2R (Primary Objective based on leaderboard).
    Optional Stretch: 1.5R only if structure is clean and does not reduce fill probability.
    Minimum Acceptable: 1.2R.
    Logic:
    TP_Distance = |Entry - SL| * RR
    LONG: Entry + TP_Distance
    SHORT: Entry - TP_Distance

# INPUT DATA
Here is the data you have access to for making your analysis:
1.  **HTF_Signal:** The latest BOS/CHoCH signal on the 1H timeframe. \n {asset_data.bos_choch_signals} \n 
Use this to determine the HTF Bias. If no signals, bias is NEUTRAL.
2.  **LTF_FVG_List:** A list of detected FVGs on the 5m timeframe (including Top, Bottom, Time, Mitigated Status). \n {asset_data.fvg_signals} \n
3. ** 5M Order Blocks:** A list of detected Order Blocks on the 5m timeframe (including Top, Bottom, Time). \n {asset_data.ob_signals} \n
4.  **Current_Price:** The latest close price on the 5m chart. \n {asset_data.current_price} \n
5. **OHLC 5M:** The recent 5-minute OHLC data for structural context. \n {asset_data.ohlc_5m} \n
6. **OHLC 1H:** The recent 1-hour OHLC data for structural context. \n {asset_data.ohlc_1h} \n

# DECISION PROCESS
1.  **Determine Bias:** specify the HTF Bias based on `HTF_Signal`.
2.  **Scan FVGs:** Filter `LTF_FVG_List` for FVGs that:
    *   Match the HTF Bias.
    *   Are `Mitigated = False`.
    *   Are structurally valid relative to `Current_Price`.
3.  **Calculate Trade:** If a valid FVG is found:
    *   Calculate Entry using the mitigation tiers above.
    *   Calculate SL using structural invalidation + buffer.
    *   Calculate TP using 1.5R (minimum 1.2R if structure blocks 1.5R).
4.  **Final Validation:** Ensure SL is not extremely tight or extremely wide.

# OUTPUT FORMAT
You must respond with a JSON object ONLY. No conversational filler.
Use the input symbol exactly (do not output "Unknown"). Use UTC timestamps for `bos_ts` and `fvg_id`.
Always include `direction` and `size` inside `parameters`. """
            + """
JSON Response Format (Strictly Follow):
{
    "decision": "ENTER_LONG" | "ENTER_SHORT" | "WAIT" | "CANCEL",
  "reasoning": "Concise explanation of HTF bias, selected FVG, and validation checks.",
  "parameters": {
        "symbol": "Ticker",
    "direction": "LONG" | "SHORT",
    "entry_price": 0.00,
    "stop_loss": 0.00,
    "take_profit": 0.00,
    "risk_reward_ratio": 1.2,
    "size": 1,
    "fvg_timestamp": "YYYY-MM-DD HH:MM:SS"},
  "metadata": {
        "htf_bias": "BULLISH" | "BEARISH",
    "confidence_score": 0-100,
    "bos_ts": "YYYY-MM-DD HH:MM:SS",
    "fvg_id": "YYYY-MM-DD HH:MM:SS"
  }
}

## Trading Rules You Follow
- Quality over quantity: only trade when HTF bias + fresh LTF FVG align.
- Structure first: never trade against HTF BOS bias.
- Risk cap: fixed $250 risk per trade (do not exceed).
- Patience: if no fresh unmitigated FVG aligned with bias, WAIT.
- Invalidation clarity: SL must be beyond structural invalidation + buffer.
- TP realism: TP must respect nearby structure; if 1.2R is blocked, WAIT.
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
