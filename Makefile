# =============================================================================
# Little Scripts Monorepo - Makefile
# =============================================================================
# Unified commands for managing all projects in the monorepo
#
# Usage:
#   make help          - Show available commands
#   make lint          - Run linter on all projects
#   make format        - Format all Python code
#   make test          - Run all tests
#   make install-all   - Install dependencies for all projects
# =============================================================================

.PHONY: help lint format test install-all install-dev clean \
        check type-check pre-commit setup \
        list-projects docker-build docker-up docker-down

# Default target
.DEFAULT_GOAL := help

# Project directories (auto-detected Python projects with requirements.txt)
PROJECTS := $(shell find . -maxdepth 2 -name "requirements.txt" -exec dirname {} \; | grep -v "^\./$$" | sort)

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m  # No Color

# =============================================================================
# Help
# =============================================================================

help:  ## Show this help message
	@echo "$(BLUE)Little Scripts Monorepo$(NC)"
	@echo "========================"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Detected projects:$(NC)"
	@for proj in $(PROJECTS); do echo "  - $$proj"; done

# =============================================================================
# Development Setup
# =============================================================================

setup: install-dev pre-commit  ## Complete development environment setup
	@echo "$(GREEN)Development environment setup complete!$(NC)"

install-dev:  ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -e ".[dev]" 2>/dev/null || pip install pre-commit pytest pytest-cov pytest-asyncio mypy ruff
	@echo "$(GREEN)Development dependencies installed.$(NC)"

pre-commit:  ## Install pre-commit hooks
	@echo "$(BLUE)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed.$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

lint:  ## Run Ruff linter on all projects
	@echo "$(BLUE)Running Ruff linter...$(NC)"
	ruff check .
	@echo "$(GREEN)Linting complete.$(NC)"

lint-fix:  ## Run Ruff linter with auto-fix
	@echo "$(BLUE)Running Ruff linter with auto-fix...$(NC)"
	ruff check --fix .
	@echo "$(GREEN)Linting with fixes complete.$(NC)"

format:  ## Format all Python code with Ruff
	@echo "$(BLUE)Formatting code with Ruff...$(NC)"
	ruff format .
	@echo "$(GREEN)Formatting complete.$(NC)"

format-check:  ## Check code formatting without making changes
	@echo "$(BLUE)Checking code format...$(NC)"
	ruff format --check .

type-check:  ## Run mypy type checker
	@echo "$(BLUE)Running mypy type checker...$(NC)"
	@for proj in $(PROJECTS); do \
		echo "$(YELLOW)Checking $$proj...$(NC)"; \
		mypy $$proj --ignore-missing-imports || true; \
	done
	@echo "$(GREEN)Type checking complete.$(NC)"

check: lint format-check type-check  ## Run all code quality checks

# =============================================================================
# Testing
# =============================================================================

test:  ## Run all tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	pytest
	@echo "$(GREEN)Tests complete.$(NC)"

test-cov:  ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-fast:  ## Run tests excluding slow tests
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest -m "not slow"

# =============================================================================
# Dependency Management
# =============================================================================

install-all:  ## Install dependencies for all projects
	@echo "$(BLUE)Installing dependencies for all projects...$(NC)"
	@for proj in $(PROJECTS); do \
		echo "$(YELLOW)Installing $$proj...$(NC)"; \
		pip install -r $$proj/requirements.txt -c constraints.txt 2>/dev/null || \
		pip install -r $$proj/requirements.txt || \
		echo "$(RED)Failed to install $$proj$(NC)"; \
	done
	@echo "$(GREEN)All project dependencies installed.$(NC)"

install-%:  ## Install dependencies for a specific project (e.g., make install-deepseek-ocr)
	@echo "$(BLUE)Installing dependencies for $*...$(NC)"
	@if [ -f "./$*/requirements.txt" ]; then \
		pip install -r ./$*/requirements.txt -c constraints.txt 2>/dev/null || \
		pip install -r ./$*/requirements.txt; \
		echo "$(GREEN)Dependencies for $* installed.$(NC)"; \
	else \
		echo "$(RED)Project $* not found or has no requirements.txt$(NC)"; \
		exit 1; \
	fi

deps-check:  ## Check for dependency conflicts across projects
	@echo "$(BLUE)Checking dependencies across projects...$(NC)"
	@echo "$(YELLOW)Common packages found in multiple projects:$(NC)"
	@cat */requirements.txt 2>/dev/null | \
		grep -v "^#" | grep -v "^$$" | grep -v "^git+" | \
		sed 's/[<>=].*//' | sed 's/\[.*//' | \
		sort | uniq -c | sort -rn | \
		awk '$$1 > 1 {print "  " $$2 " (in " $$1 " projects)"}'

deps-tree:  ## Show dependency tree for all projects
	@echo "$(BLUE)Dependency overview:$(NC)"
	@for proj in $(PROJECTS); do \
		echo "\n$(YELLOW)$$proj:$(NC)"; \
		cat $$proj/requirements.txt 2>/dev/null | grep -v "^#" | grep -v "^$$" | sed 's/^/  /'; \
	done

# =============================================================================
# Docker Operations
# =============================================================================

docker-build:  ## Build Docker images for all projects with Dockerfiles
	@echo "$(BLUE)Building Docker images...$(NC)"
	@for proj in $(PROJECTS); do \
		if [ -f "$$proj/Dockerfile" ]; then \
			echo "$(YELLOW)Building $$proj...$(NC)"; \
			docker build -t little-scripts/$$(basename $$proj) $$proj; \
		fi \
	done
	@echo "$(GREEN)Docker builds complete.$(NC)"

docker-up:  ## Start all Docker Compose services
	@echo "$(BLUE)Starting Docker services...$(NC)"
	@for proj in $(PROJECTS); do \
		if [ -f "$$proj/docker-compose.yml" ]; then \
			echo "$(YELLOW)Starting $$proj...$(NC)"; \
			docker-compose -f $$proj/docker-compose.yml up -d; \
		fi \
	done
	@echo "$(GREEN)Services started.$(NC)"

docker-down:  ## Stop all Docker Compose services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	@for proj in $(PROJECTS); do \
		if [ -f "$$proj/docker-compose.yml" ]; then \
			echo "$(YELLOW)Stopping $$proj...$(NC)"; \
			docker-compose -f $$proj/docker-compose.yml down; \
		fi \
	done
	@echo "$(GREEN)Services stopped.$(NC)"

# =============================================================================
# Cleanup
# =============================================================================

clean:  ## Remove build artifacts and cache files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete.$(NC)"

clean-venv:  ## Remove all virtual environments
	@echo "$(BLUE)Removing virtual environments...$(NC)"
	find . -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "venv" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "env" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Virtual environments removed.$(NC)"

# =============================================================================
# Utilities
# =============================================================================

list-projects:  ## List all detected projects
	@echo "$(BLUE)Projects in monorepo:$(NC)"
	@for proj in $(PROJECTS); do \
		echo "  $(GREEN)$$proj$(NC)"; \
		if [ -f "$$proj/README.md" ]; then \
			head -n 3 $$proj/README.md | grep -E "^#" | sed 's/^/    /'; \
		fi \
	done

update-pre-commit:  ## Update pre-commit hooks to latest versions
	@echo "$(BLUE)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate
	@echo "$(GREEN)Pre-commit hooks updated.$(NC)"

run-pre-commit:  ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit on all files...$(NC)"
	pre-commit run --all-files
