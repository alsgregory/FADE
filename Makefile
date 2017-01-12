all:

lint:
	@echo "    Linting fade codebase"
	@flake8 fade
	@echo "    Linting fade test suite"
	@flake8 tests
	@echo "    Linting fade demo suite"
	@flake8 examples
	@echo "    Linting fade tutorial suite"
	@flake8 tutorials

test:
	@echo "    Running all tests"
	@py.test tests $(PYTEST_ARGS)
