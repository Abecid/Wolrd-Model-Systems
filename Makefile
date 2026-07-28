.PHONY: install test lint format models bootstrap report service

install:
	python -m pip install -e '.[dev,gpu,service]'

test:
	pytest -q

lint:
	ruff check src tests examples

format:
	ruff format src tests examples

models:
	wms models

bootstrap:
	bash scripts/bootstrap_model.sh $${MODEL:?set MODEL}

report:
	wms-report $${RUN_DIR:?set RUN_DIR}

service:
	wms-serve --config configs/service.yaml
