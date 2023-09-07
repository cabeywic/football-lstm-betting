init:
	pip install -r requirements.txt

test:
	pytest -r f

docs:
	sphinx-build -b html docs/ docs/_build

help:
	@echo "Available targets:"
	@echo "init - Install project dependencies"
	@echo "test - Run test cases"
	@echo "docs - Generate HTML documentation"

.PHONY: init test docs help
