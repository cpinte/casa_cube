# Casa Cube Documentation

This directory contains the documentation for the Casa Cube package.

## Building the Documentation

To build the documentation locally:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the documentation:
   ```bash
   cd docs
   make html
   ```

3. View the documentation:
   Open `docs/_build/html/index.html` in your web browser.

## Documentation Structure

- `index.rst`: Main documentation page
- `installation.rst`: Installation instructions
- `usage.rst`: Usage guide with examples
- `api.rst`: API reference
- `examples.rst`: Example use cases

## Contributing to the Documentation

1. Edit the relevant .rst files in this directory
2. Rebuild the documentation to check your changes
3. Submit a pull request with your changes

## ReadTheDocs

The documentation is automatically built and hosted on ReadTheDocs when changes are pushed to the main branch. 