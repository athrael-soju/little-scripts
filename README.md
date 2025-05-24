# Python Monorepo

A simple Python monorepo structure for managing multiple packages and applications, optimized for use with [UV](https://github.com/astral-sh/uv) - the fast Python package installer.

## Structure

```
├── packages/           # Shared libraries and utilities
│   ├── utils/         # Common utilities package
│   └── core/          # Core business logic package
├── apps/              # Applications
│   ├── web-app/       # Web application
│   └── cli-tool/      # Command line tool
├── tests/             # Shared tests
├── docs/              # Documentation
├── scripts/           # Build and utility scripts
├── pyproject.toml     # Root project configuration
├── requirements.txt   # Common dependencies
└── README.md          # This file
```

## Getting Started

### Quick Setup (Recommended)

Run the automated setup script:
```bash
python scripts/setup.py
```

This will:
- Install UV if not already present
- Install all dependencies using UV
- Set up all packages in development mode

### Manual Setup

If you prefer to set up manually:

1. Install UV (if not already installed):
   ```bash
   pip install uv
   # or visit: https://github.com/astral-sh/uv
   ```

2. Install dependencies with UV:
   ```bash
   uv pip install -r requirements.txt
   uv pip install -e .[dev]
   uv pip install -e packages/utils
   uv pip install -e packages/core
   uv pip install -e apps/web-app
   uv pip install -e apps/cli-tool
   ```

3. Run tests:
   ```bash
   python -m pytest tests/
   ```

## Why UV?

This monorepo is optimized for [UV](https://github.com/astral-sh/uv), which provides:

- ⚡ **10-100x faster** than pip
- 🔒 **Reliable dependency resolution**
- 🛠️ **Drop-in replacement** for pip
- 📦 **Better caching** and parallelization

## Development

Each package and app has its own `setup.py` for independent development while sharing common dependencies through the root configuration.

### Using UV for Development

- **Add new dependency**: `uv add <package>`
- **Install package**: `uv pip install <package>`
- **Create lock file**: `uv lock`
- **Sync dependencies**: `uv pip sync requirements.txt`

### Package Management

- Use `-e` flag when installing local packages for development
- Shared dependencies are managed in the root `requirements.txt`
- Package-specific dependencies are in each package's setup file 