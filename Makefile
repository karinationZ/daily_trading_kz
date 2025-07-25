run_order_block_sweep_bot:
	@echo "Starting OrderBlocks Sweep Bot"
	poetry run python3 daily_trading/bots/smc_bot/run_order_block_sweep_bot.py

run_bos_fvg_tap_bot:
	@echo "Starting BOS & FVG Tap Bot"
	poetry run python3 daily_trading/bots/smc_bot/run_bos_fvg_tap_bot.py

run_bos_ob_tap_bot:
	@echo "Starting BOS & Order Block Tap Bot"
	poetry run python3 daily_trading/bots/smc_bot/run_bos_order_block_tap_bot.py