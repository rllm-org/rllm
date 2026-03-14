# rLLM Documentation

This directory contains the documentation for the rLLM library, built using [MkDocs](https://www.mkdocs.org/) with [Material theme](https://squidfunk.github.io/mkdocs-material/) and [mkdocstrings](https://mkdocstrings.github.io/) for API documentation.

## 🚀 Quick Start

### Building the Documentation

To build the documentation:

```bash
./build_docs.sh
```

### Serving the Documentation Locally

To build and serve the documentation with live reload:

```bash
./build_docs.sh serve
```

The documentation will be available at `http://localhost:8000`.

## 📁 Structure

```
docs/
├── mkdocs.yml          # MkDocs configuration
├── build_docs.sh      # Build script
├── docs/              # Documentation content
│   ├── index.md       # Homepage
│   ├── api/           # API documentation (generated)
│   ├── examples/      # Example guides
│   ├── getting-started/ # Getting started guides
│   └── core-concepts/ # Core concept explanations
|   └── experimental/  # Experimental features
└── site/              # Generated static site (after build)
```

## 🔧 Features

### API Documentation
- **Automatic API docs**: Generated from docstrings using mkdocstrings
- **Google-style docstrings**: Supports Google-style docstring format
- **Source code links**: Direct links to source code
- **Type hints**: Shows function signatures and type annotations

### Documentation Features
- **Material Design**: Modern, responsive theme
- **Code highlighting**: Syntax highlighting for multiple languages
- **Navigation**: Automatic navigation generation
- **Search**: Full-text search functionality
- **Mobile-friendly**: Responsive design for all devices

## ✍️ Writing Documentation

### Adding New Pages
1. Create a new `.md` file in the appropriate `docs/` subdirectory
2. Add the page to the `nav` section in `mkdocs.yml`
3. Use Markdown syntax for content

### API Documentation
API documentation is automatically generated from Python docstrings using mkdocstrings. To document a new module:

1. Add a new file in `docs/api/`
2. Use the mkdocstrings syntax: `::: module.name`
3. Add the page to the navigation in `mkdocs.yml`

Example:
```markdown
# My Module

Brief description of the module.

::: rllm.my_module
```

### Code Examples
Use fenced code blocks with language specification:

```python
from rllm.agents import Agent

agent = Agent()
```

## 🛠️ Customization

### Theme Configuration
The Material theme is configured in `mkdocs.yml`. You can customize:
- Colors and palette
- Navigation features
- Extensions and agenthub

### Extensions
Currently enabled extensions:
- `pymdownx.highlight`: Code highlighting
- `pymdownx.superfences`: Enhanced code blocks
- `admonition`: Call-out boxes
- `pymdownx.details`: Collapsible sections

## 📝 Dependencies

Documentation dependencies are automatically installed with the main package:
- `mkdocs`: Static site generator
- `mkdocs-material`: Material Design theme
- `mkdocstrings[python]`: API documentation from docstrings
- `mkdocs-autorefs`: Cross-references
- `pymdown-extensions`: Enhanced Markdown extensions

## 🐛 Troubleshooting

### Common Issues

**Import errors when building**:
- Ensure the rLLM package is properly installed: `uv pip install -e ..`
- Check that all dependencies are available

**Missing API documentation**:
- Verify the module path in the mkdocstrings directive
- Check that the module has proper docstrings

**Build fails**:
- Check that all dependencies are installed: `uv pip install -e .`
- Verify that the `mkdocs.yml` syntax is correct

### Getting Help
- Check the [MkDocs documentation](https://www.mkdocs.org/)
- Review [mkdocstrings documentation](https://mkdocstrings.github.io/)
- Open an issue in the rLLM repository 