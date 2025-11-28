
PYTHON_VERSION := 3.13.0

create-env:
	@echo "Creating virtual environment with uv..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Error: uv is not installed. Please install uv first."; \
		exit 1; \
	fi
	@echo "Creating virtual environment with Python ${PYTHON_VERSION}..."
	@uv venv --python ${PYTHON_VERSION}
	@echo "Virtual environment created successfully!"
