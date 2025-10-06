# Contributing to Zynkode Compute

## Development Setup

```bash
git clone https://github.com/Zara-Quinn/zynkode-compute.git
cd zynkode-compute
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Code Style

- Follow PEP 8
- All public methods must have docstrings
- Type hints required for all function signatures
