.PHONY: all install lint test format

all: lint test

install:
	pip install -r requirements.dev.txt  -e .

compile:
	pip-compile requirements.in; pip-compile requirements.dev.in

upgrade:
	pip-compile --upgrade requirements.in; pip-compile --upgrade requirements.dev.in

lint:
	flake8 .
	pydocstyle truman
	mypy truman
	black --check .


test:
	pytest tests

format:
	black .
