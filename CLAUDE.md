# CLAUDE.md - Guide for Marine Navigation USV Mission Planning

## Project Configuration
- **Environment**: Python with UV packaging
- **Data**: Nautical chart data in TXT and compressed file formats
- **Visualization**: QGIS (sample-ausenc.qgz)

## Commands
- Setup: `uv venv && uv pip install -r requirements.txt`
- Run: `uv run main.py`
- Run examples: `uv run examples/path_planning_demo.py`
- Tests: `uv run -m pytest` or `uv run -m pytest tests/test_specific.py -v`
- Lint: `uv run -m ruff check .`
- Format: `uv run -m black .`

## Code Guidelines
- **Imports**: Group standard library, third-party, and local imports
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Types**: Use type hints for all functions and variables
- **Documentation**: Docstrings for all functions and classes (Google style)
- **Error Handling**: Use try/except with specific exceptions
- **Algorithms**: Implement search patterns from README.md
- **Data Processing**: Handle nautical chart data with appropriate libraries