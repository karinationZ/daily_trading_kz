POETRY_RUN = poetry run

lint:
	@echo "Running linting..."
	$(POETRY_RUN) black . && $(POETRY_RUN) autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive . && $(POETRY_RUN) flake8 .

run_ibkr_bot:
	@echo "Starting IBKR Master Bot..."
	poetry run python3 daily_trading/bots/smc_bot/run_ibkr_master_bot.py