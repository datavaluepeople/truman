.PHONY: install lint test format

install:
	pip install -r requirements.dev.txt  -e .

lint:
	flake8 .
	black --check .

test:
	pytest tests

format:
	black .
