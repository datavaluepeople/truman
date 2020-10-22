.PHONY: all install lint test format

all: lint test

install:
	pip install -r requirements.dev.txt  -e .

lint:
	flake8 .
	mypy truman
	black --check .


test:
	pytest tests

format:
	black .
