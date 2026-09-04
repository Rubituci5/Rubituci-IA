# Entity Project - Makefile
# Common development tasks

.PHONY: help install test lint format type-check db-migrate db-upgrade db-downgrade docker-up docker-down docker-logs clean

# Default target
help:
	@echo "Entity Project - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install Python and Node dependencies"
	@echo "  install-python   Install Python dependencies only"
	@echo "  install-node     Install Node dependencies only"
	@echo ""
	@echo "Development:"
	@echo "  test             Run all Python tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-frontend    Run frontend tests"
	@echo "  lint             Run all linters"
	@echo "  format           Format code (ruff + prettier)"
	@echo "  type-check       Run type checkers (mypy + tsc)"
	@echo ""
	@echo "Database:"
	@echo "  db-migrate       Create new migration"
	@echo "  db-upgrade       Apply migrations"
	@echo "  db-downgrade     Revert last migration"
	@echo "  db-reset         Drop and recreate database"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up        Start all services"
	@echo "  docker-down      Stop all services"
	@echo "  docker-logs      View logs"
	@echo "  docker-build     Build all images"
	@echo ""
	@echo "Entity:"
	@echo "  init             Initialize entity (create Generation 000001 snapshot)"
	@echo "  train            Train current generation"
	@echo "  train-gen        Train specific generation (usage: make train-gen GEN=2)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Clean build artifacts"
	@echo "  clean-all        Clean everything including venv and node_modules"

# ============================================
# INSTALL
# ============================================
install: install-python install-node

install-python:
	pip install -e .[dev]
	pre-commit install

install-node:
	cd web && npm install

# ============================================
# TESTS
# ============================================
test:
	python run_tests.py

test-unit:
	python -m pytest tests/test_brain.py tests/test_memory.py tests/test_research.py tests/test_reflection_consolidation.py tests/test_evolution_training.py tests/test_api_security.py -v

test-integration:
	python -m pytest tests/test_integration.py -v

test-frontend:
	cd web && npm test

test-coverage:
	python -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# ============================================
# LINT & FORMAT
# ============================================
lint:
	ruff check .
	cd web && npm run lint

format:
	ruff format .
	cd web && npm run format

type-check:
	mypy .
	cd web && npm run type-check

# ============================================
# DATABASE
# ============================================
db-migrate:
	alembic revision --autogenerate -m "$(MSG)"

db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-reset:
	alembic downgrade base
	alembic upgrade head

db-shell:
	psql $(DATABASE_URL)

# ============================================
# DOCKER
# ============================================
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

docker-restart:
	docker-compose restart

docker-ps:
	docker-compose ps

# ============================================
# ENTITY OPERATIONS
# ============================================
init:
	python scripts/init_entity.py

train:
	python scripts/train_generation.py 1

train-gen:
	python scripts/train_generation.py $(GEN)

# ============================================
# CLEANUP
# ============================================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/

clean-all: clean
	rm -rf .venv/ venv/ env/
	cd web && rm -rf node_modules/ .next/ out/

# ============================================
# DEVELOPMENT HELPERS
# ============================================
run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-worker:
	celery -A api.celery_app worker --loglevel=info

run-beat:
	celery -A api.celery_app beat --loglevel=info

run-web:
	cd web && npm run dev

# Run all services locally (requires postgres, redis running)
run-all: run-api run-worker run-beat run-web

# ============================================
# CHECKS
# ============================================
check: lint type-check test
	@echo "All checks passed!"

pre-commit:
	pre-commit run --all-files

# ============================================
# DOCUMENTATION
# ============================================
docs:
	@echo "Documentation available in README.md"
	@echo "API docs at http://localhost:8000/docs when running"