# Strategy Overview: Institutional ORB Pro

This strategy is designed for professional intraday trading on 5-minute charts. It does not "chase" the market. Instead, it identifies where big institutions have entered the market at the open and waits for the price to return to those specific levels before entering.

---

### **Phase 1: Establishing the Daily Direction (The ORB)**
*   **The 15-Minute Box:** The bot monitors the first 15 minutes of the trading session to create a "price box."
*   **The Breakout:** We wait for a 5-minute candle to close above or below this box. 
    *   If it closes **above**, our bias is **LONG** (Buy only).
    *   If it closes **below**, our bias is **SHORT** (Sell only).

### **Phase 2: Finding the Institutional Footprint (SMC Zones)**
Once we have a direction, the bot scans for **Supply and Demand zones** (Order Blocks).
*   **Imbalance Requirement:** Every zone must be followed by a "Fair Value Gap" (a big jump in price). This proves that the move was caused by institutions, not retail traders.
*   **Freshness:** The bot only looks at zones created 1 hour before the open or during the current session.
*   **Context:** The zone must be "supported." It cannot be floating in middle of nowhere; it must be near the Opening Range lines or a major historical turning point.

### **Phase 3: The Trade Trigger (The Retrace)**
*   **Wait for the Retest:** We do not enter when the price is breaking out. We wait for the price to come back ("retrace") and touch one of our identified zones.
*   **Proximity Selection:** If multiple zones exist, the bot automatically targets the one **closest** to the current market price.

### **Phase 4: Confirmation (The Absorption Check)**
When the price touches a zone, the bot performs a "Safety Check" called **Absorption**:
*   **Rejection Wicks:** The bot looks for a "wick" on the candle. This shows that the zone is working like a floor (for Longs) or a ceiling (for Shorts) and that the price is bouncing.
*   **No "Falling Knives":** If the price is slamming through the zone with a giant body and high volume (no wick), the bot cancels the trade. We only enter if we see the market "absorbing" the pressure.

### **Phase 5: Execution Rules**
The bot treats the two types of zones differently:
1.  **Swing Zones (Major Turns):** These are very strong. The bot is **Aggressive** and enters at the "Market Price" the moment a 5-minute candle touches the zone and shows a bounce.
2.  **Internal Zones (Minor Turns):** These are tactical. The bot is **Conservative** and sets a "Limit Order" at the very edge of the zone to get the best possible price.

### **Phase 6: Risk & Safety**
*   **Fixed Risk:** Every trade is calculated to lose exactly **$150** if the Stop Loss is hit. The bot automatically decides how many contracts to trade based on the distance to the Stop Loss.
*   **Take Profit:** The bot aims for a **1.5x return**. If we risk $150, we aim to make $225.
*   **Session Close:** 15 minutes before the market officially closes, the bot cancels all pending orders and closes any open trades. We never hold positions overnight.
*   **Automatic Cancellation:** If the price hits our "Profit Target" before our entry is touched, the trade is cancelled. We don't "chase" a move that has already finished.