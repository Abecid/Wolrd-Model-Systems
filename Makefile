.PHONY: install test lint format bootstrap report service

install:
	python -m pip install -e '.[dev,gpu,service]'

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

bootstrap:
	bash scripts/bootstrap_upstream.sh

report:
	mgs-report $${RUN_DIR:?set RUN_DIR}

service:
	mgs-serve --config configs/service.yaml
