# Little Scripts

A monorepo for small AI-powered Python scripts. Each script lives in its own directory under `scripts/` with independent functionality while sharing common dependencies and utilities.

## Structure

```
little-scripts/
├── scripts/                    # Individual AI scripts
│   ├── script-name/           # Each script in its own folder
│   │   ├── main.py           # Script entry point
│   │   ├── requirements.txt  # Script-specific dependencies
│   │   └── README.md         # Script documentation
│   └── ...
├── shared/                    # Shared utilities and common code
│   ├── __init__.py
│   ├── utils.py              # Common utility functions
│   └── ai_helpers.py         # Shared AI/ML helper functions
├── requirements.txt          # Global dependencies
├── pyproject.toml           # Python project configuration
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## Quick Start

1. **Setup virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install global dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a new script:**
   ```bash
   mkdir scripts/my-new-script
   cd scripts/my-new-script
   # Create your script files
   ```

4. **Run a script:**
   ```bash
   cd scripts/your-script-name
   python main.py
   ```

## Adding New Scripts

Each script should be self-contained in its own directory under `scripts/`. Include:
- `main.py` - Entry point for your script
- `requirements.txt` - Any additional dependencies specific to this script
- `README.md` - Documentation for what the script does and how to use it

## Shared Utilities

Common functionality used across multiple scripts should go in the `shared/` directory. Import them in your scripts like:

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.utils import some_utility_function
from shared.ai_helpers import some_ai_helper
``` 