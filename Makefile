about:
	@echo "PipeStonks project"

SRC_DIRS := # placeholder  

.PHONY: format
format:
	python -m black $(SRC_DIRS)

.PHONY: check-format
check-format:
	python -m black --check $(SRC_DIRS)

.PHONY: lint
lint:
	python -m flake8 $(SRC_DIRS)

.PHONY: type
type:
	mypy $(SRC_DIRS)

.PHONY: isort
isort:
	isort $(SRC_DIRS)

.PHONY: pre-commit
pre-commit: isort format lint type 

.PHONY: pre-commit-safe
pre-commit-safe: check-format lint
