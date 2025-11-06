.PHONY: setup run clean help

# Default target
help:
	@echo "Bandana Snip Engine - Make Commands"
	@echo "==================================="
	@echo "setup  - Set up virtual environment and install dependencies"
	@echo "run    - Run the application (after setup)"
	@echo "clean  - Remove virtual environment and cache files"
	@echo "help   - Show this help message"

setup:
	@echo "🔧 Setting up Bandana Snip Engine..."
	./start_bandana.sh

run:
	@echo "▶ Running Bandana Snip Engine..."
	./run_local.sh

clean:
	@echo "🧹 Cleaning up..."
	rm -rf bandana_snip_engine/.venv
	rm -rf venv venv_gui venv_gui_fixed
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"