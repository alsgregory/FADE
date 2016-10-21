all:

lint:
	@echo "    Linting firedrake_da codebase"
	@flake8 firedrake_da
	@echo "    Linting firedrake_da test suite"
	@flake8 tests
	@echo "    Linting firedrake_da demo suite"
	@flake8 examples

test:
	@echo "    Running all tests"
	@py.test tests $(PYTEST_ARGS)
