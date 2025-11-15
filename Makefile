########################################################################################################################
# Project installation
########################################################################################################################

install:
	uv sync

########################################################################################################################
# Quality checks
########################################################################################################################

test:
	uv run pytest tests --cov src --cov-report term --cov-report=html --cov-report xml --junit-xml=tests-results.xml

format-check:
	uv run ruff format --check src tests

format-fix:
	uv run ruff format src tests

lint-check:
	uv run ruff check src tests

lint-fix:
	uv run ruff check src tests --fix

type-check:
	uv run mypy src

########################################################################################################################
# Compression
########################################################################################################################

compress-repo:
	tar cJvf projet1.tar.xz --exclude='.venv' --exclude='__pycache__' --exclude='*/__pycache__' --exclude='matieres' --exclude='.env' --exclude='.git' --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='projet1.tar.xz' .
