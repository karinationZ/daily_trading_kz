POETRY_RUN = poetry run

lint:
	@echo "Running linting..."
	$(POETRY_RUN) black . && $(POETRY_RUN) autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive . && $(POETRY_RUN) flake8 .

run_order_block_sweep_bot:
	@echo "Starting OrderBlocks Sweep Bot"
	poetry run python3 daily_trading/bots/smc_bot/run_order_block_sweep_bot.py

run_bos_15m_fvg_tap_bot:
	@echo "Starting BOS & FVG Tap Bot"
	poetry run python3 daily_trading/bots/smc_bot/run_bos_15m_fvg_tap_bot.py

run_bos_15m_ob_tap_bot:
	@echo "Starting BOS & Order Block Tap Bot"
	poetry run python3 daily_trading/bots/smc_bot/run_bos_15m_ob_tap_bot.py

run_support_resistance_bot:
	@echo "Starting Support & Resistance Bot"
	poetry run python3 daily_trading/bots/smc_bot/run_support_resistance_bot.py

run_ibkr_bot:
	@echo "Starting Master Bot..."
	poetry run python3 daily_trading/bots/smc_bot/run_ibkr_master_bot.py