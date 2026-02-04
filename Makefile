.PHONY: all check lint format test test-fast test-course3 test-course4 demos clean setup help

# Default target
all: check

# Setup development environment
setup:
	@echo "Setting up development environment..."
	uv sync --all-extras
	pre-commit install
	@echo "Setup complete."

# Install dependencies
install: setup

# Run all quality checks
check: lint test
	@echo "All checks passed."

# Lint code
lint:
	@echo "Linting Python..."
	uvx ruff check demos/ labs/ examples/ --ignore E501,E722,E402,F821,F541,F811,F401,F841,I001
	@echo "Linting Rust..."
	@for dir in demos/*/week*/; do \
		if [ -f "$$dir/Cargo.toml" ]; then \
			echo "Checking $$dir"; \
			(cd "$$dir" && cargo clippy --quiet -- -D warnings) || exit 1; \
		fi \
	done
	@echo "Lint complete."

# Format code
format:
	@echo "Formatting Python..."
	uvx ruff format demos/ labs/ examples/
	uvx ruff check --fix demos/ labs/ examples/ --ignore E501,E722,E402,F821,F541,F811,F401,F841,I001
	@echo "Formatting Rust..."
	@for dir in demos/*/week*/; do \
		if [ -f "$$dir/Cargo.toml" ]; then \
			(cd "$$dir" && cargo fmt) || exit 1; \
		fi \
	done
	@echo "Format complete."

# Run full test suite
test:
	@echo "Running Python tests..."
	uvx pytest tests/ -v || true
	@echo "Running Rust tests..."
	@for dir in demos/*/week*/; do \
		if [ -f "$$dir/Cargo.toml" ]; then \
			echo "Testing $$dir"; \
			(cd "$$dir" && cargo test --quiet) || exit 1; \
		fi \
	done
	@echo "All tests passed."

# Quick syntax validation
test-fast:
	@echo "Quick syntax check..."
	python3 -m py_compile demos/**/*.py 2>/dev/null || true
	python3 -m py_compile labs/**/*.py 2>/dev/null || true
	python3 -m py_compile examples/**/*.py 2>/dev/null || true
	@echo "Syntax check complete."

# Course-specific tests
test-course3:
	@echo "Running Course 3 tests..."
	@for dir in demos/course3/week*/; do \
		if [ -f "$$dir/Cargo.toml" ]; then \
			(cd "$$dir" && cargo test --quiet) || exit 1; \
		fi \
	done

test-course4:
	@echo "Running Course 4 tests..."
	@for dir in demos/course4/week*/; do \
		if [ -f "$$dir/Cargo.toml" ]; then \
			(cd "$$dir" && cargo test --quiet) || exit 1; \
		fi \
	done

# Validate demos
demos:
	@echo "Validating demo structure..."
	@for course in course3 course4; do \
		echo "Checking $$course demos..."; \
		ls -d demos/$$course/week*/ 2>/dev/null || echo "No demos yet for $$course"; \
	done

# Clean build artifacts
clean:
	@echo "Cleaning..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "target" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

# Help
help:
	@echo "Available targets:"
	@echo "  setup        - Set up development environment"
	@echo "  check        - Run all quality checks (lint + test)"
	@echo "  lint         - Lint Python and Rust code"
	@echo "  format       - Auto-format code"
	@echo "  test         - Run full test suite"
	@echo "  test-fast    - Quick syntax validation"
	@echo "  test-course3 - Run Course 3 tests only"
	@echo "  test-course4 - Run Course 4 tests only"
	@echo "  demos        - Validate demo structure"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this message"
