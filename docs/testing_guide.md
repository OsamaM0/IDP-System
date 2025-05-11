# IDP-System Testing Guide

This document outlines the testing strategy for the IDP-System project and provides instructions for running and writing tests.

## Testing Structure

The IDP-System testing suite is organized as follows:

- **Unit Tests**: Test individual components in isolation
  - BoundingBox functionality
  - OCR engines
  - Controllers 
  - Data processing utilities

- **Integration Tests**: Test interaction between components
  - API endpoints
  - End-to-end document processing workflows

## Running Tests

You can use the `run_tests.py` script to run the test suite:

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit-only

# Run only integration tests
python run_tests.py --integration-only

# Run with increased verbosity
python run_tests.py -v

# Generate test coverage report
python run_tests.py --coverage
```

## Creating New Tests

### For Unit Tests

1. Create a new file in the appropriate directory under `tests/unit/`
2. Follow the naming convention: `test_*.py`
3. Create test classes and methods that use pytest fixtures and assertions

Example:

```python
import pytest
from your_module import YourClass

class TestYourClass:
    def test_some_functionality(self):
        instance = YourClass()
        result = instance.some_method()
        assert result == expected_value
```

### For Integration Tests

1. Create a new file under `tests/integration/`
2. Use the TestClient from FastAPI to test API endpoints
3. Mock external dependencies as needed

Example:

```python
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_endpoint(client):
    response = client.get("/some-endpoint")
    assert response.status_code == 200
    assert response.json() == expected_response
```

## Test Coverage

To check test coverage, run:

```bash
python run_tests.py --coverage
```

This will generate a coverage report showing which parts of the codebase are covered by tests. The HTML report will be available in the `htmlcov` directory.

## Continuous Integration

Tests are automatically run on every pull request through our CI pipeline. Pull requests cannot be merged unless all tests pass.

## Best Practices

- Keep tests independent and isolated
- Use meaningful test names that describe what is being tested
- Test both positive and negative scenarios
- Use fixtures for common setup code
- Mock external dependencies to ensure unit tests run quickly
