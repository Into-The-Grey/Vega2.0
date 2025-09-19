# Federated Learning Tests

This directory contains all unit tests for the federated learning components of Vega 2.0.

## Test Structure

- `test_security.py` - Security module tests (authentication, signatures, anomaly detection)
- `test_participant.py` - Participant module tests (registration, training, communication)
- `test_communication.py` - Communication module tests (messaging, networking, protocols)
- `test_*.py` - Additional component-specific tests

## Running Tests

From the project root:

```bash
# Run all federated tests
python -m pytest tests/federated/ -v

# Run specific test file
python -m pytest tests/federated/test_security.py -v

# Run with coverage
python -m pytest tests/federated/ --cov=src.vega.federated
```