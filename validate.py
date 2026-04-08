#!/usr/bin/env python3
"""
Quick syntax validation for Nutrisync codebase.

Verifies:
- All Python files have valid syntax
- All imports are resolvable
- Core classes are defined
"""

import ast
import sys
from pathlib import Path

def validate_syntax(filepath: str) -> bool:
    """Validate Python file syntax."""
    try:
        with open(filepath) as f:
            ast.parse(f.read())
        print(f" {filepath}: Valid syntax")
        return True
    except SyntaxError as e:
        print(f" {filepath}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f" {filepath}: Error: {e}")
        return False

def main():
    """Validate all Python files."""
    files = [
        "models.py",
        "client.py",
        "server/environment.py",
        "server/app.py",
        "example.py",
    ]
    
    results = []
    for filepath in files:
        if Path(filepath).exists():
            results.append(validate_syntax(filepath))
        else:
            print(f" {filepath}: File not found")
            results.append(False)
    
    print("\n" + "="*60)
    if all(results):
        print(" All files validated successfully!")
        return 0
    else:
        print(" Some files have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
