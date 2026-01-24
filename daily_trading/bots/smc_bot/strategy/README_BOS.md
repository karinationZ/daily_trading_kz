# 🏛️ Institutional Structure Pro (BOS-Limit)

## Overview

**Institutional Structure Pro (BOS-Limit)** is a trend-following **Smart Money Concept (SMC)** trading strategy designed for **Limit Order entries**.

Unlike breakout bots that chase price, this strategy is **passive**.
It:

1. Identifies a **Break of Structure (BOS)** on a higher timeframe
2. Finds the **Supply/Demand zone** responsible for that break
3. Places a **Limit Order deep inside the zone**, waiting for price to retrace for a high-precision entry

This mirrors how institutional traders seek **optimal entry, minimal drawdown, and structural confirmation**.

---

## 🧠 Core Logic

The strategy operates in **two distinct phases**.

---

### Phase 1: Structural Bias (HTF)

* **Detector:** `BosChochDetector`
* **Timeframe:** Main timeframe (default: `1h`)
* **Logic:**

  * Detects the most recent **Break of Structure (BOS)**
  * Determines **directional bias** (Long or Short)

**Outcome:**

* ✅ Bias = Long / Short → proceed
* ❌ Bias = Neutral → bot remains idle

---

### Phase 2: Sniper Entry (Zone & Limit)

* **Detector:** `SupplyDemandZonesDetector`
* **Logic:**

  * Identifies **Swing** and **Internal** Supply/Demand zones aligned with bias
  * Selects the **freshest, unmitigated zone**
  * Calculates a **Limit Order price** using `entry_depth`
  * Places the order and waits for retrace

---

## 🔄 Trade Lifecycle

### 1. Scan

* Continuously scans for **1H BOS**

### 2. Pending

* Valid zone found → places **Bracket Limit Order**
* **Status:** `PENDING`

### 3. Management

* **Fill:** Price retraces → order filled

  * **Status:** `FILLED`
* **Cancel (Time):** Session ends → order cancelled
* **Cancel (Structure):** New BOS occurs before fill → order invalidated

### 4. Exit

* **Passive Exit:** TP or SL hit
* **Forced Exit:** Hard cutoff at session close (e.g., NY Close)

---

## ⚙️ Key Configuration Parameters

| Parameter        | Type    | Default | Description                                                  |
| ---------------- | ------- | ------- | ------------------------------------------------------------ |
| `main_tf`        | `str`   | `"1h"`  | Timeframe used for BOS & zone detection                      |
| `entry_depth`    | `float` | `0.5`   | Depth into zone for limit entry (0.0 = edge, 0.5 = midpoint) |
| `risk`           | `float` | `150.0` | Dollar risk per trade (used for position sizing)             |
| `tp_multiple`    | `float` | `1.5`   | Risk-to-Reward ratio                                         |
| `sl_buffer_mult` | `float` | `0.1`   | ATR multiplier added beyond zone for SL padding              |
| `priority`       | `int`   | `1`     | Low priority (can be overridden by faster strategies)        |

---

## 🛡️ Safety & Concurrency — *The “Bouncer” System*

This strategy inherits from `SMCBotBase` and uses a **Priority Lock System** to manage symbol access.

### Key Rules

* **Non-Blocking:** Checks if a symbol is already locked
* **Low Priority:** Priority = `1`
* **Override Logic:**

  * Higher-priority strategies (e.g., ORB, Priority `2`) can override **pending** BOS orders
* **Protection:**

  * If BOS is already **FILLED**, it holds the lock and **cannot be interrupted**

---

## 📊 Exit Types

| Exit Type         | Description                                        |
| ----------------- | -------------------------------------------------- |
| **WIN / LOSS**    | Standard TP or SL hit                              |
| **CUTOFF**        | Trading session ended                              |
|                   | • Pending → order cancelled                        |
|                   | • Filled → position closed at market               |
| **NEW_STRUCTURE** | New BOS occurred while waiting → setup invalidated |

---

## 🚦 The “Bouncer” Logic Explained

This system prevents multiple bots from fighting over the same symbol while prioritizing **fast strategies (ORB)** over **slow strategies (BOS)**.

---

### 🔐 The Lock (`LockInfo`)

A digital tag attached to a symbol (e.g., `MNQ`) containing:

* **Owner:** Strategy tag
* **State:** `PENDING` or `FILLED`
* **Rank:** Priority level
* **Timestamp:** Acquisition time

---

### 🚪 Acquisition Logic (“The Bouncer”)

**Case A — Empty Room**

* No lock exists
* ✅ Lock created → entry allowed

**Case B — Self**

* Same strategy already holds the lock
* ✅ No action → continue

**Case C — Override**

* Another strategy is **PENDING**
* Current strategy has **higher priority**
* 🔄 Action:

  * Cancel opponent’s IBKR orders
  * Wait for flat confirmation (`wait_for_flat`)
  * Overwrite lock
* ✅ Entry allowed

**Case D — Blocked**

* Another strategy is **FILLED**
* ❌ Access denied — cannot interrupt live trade

### 🧹 Cleanup (“The Janitor”)

At the end of every cycle (success, failure, or crash):

* **If Strategy = `NONE`**

  * Lock is destroyed
* **If Strategy = `PENDING` or `FILLED`**

  * Lock is renewed or updated

This guarantees **clean state, crash safety, and no ghost locks**.

