#!/bin/bash
# Upload script for clap-llm package to PyPI

set -e  # Exit on error

echo "Uploading clap-llm to PyPI..."

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    echo "Installing twine..."
    pip install twine
fi

# Check if build is installed
if ! command -v python -m build &> /dev/null; then
    echo "Installing build..."
    pip install build
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Build the package
echo "Building package..."
python -m build

# Check if build was successful
if [ ! -d "dist" ]; then
    echo "Error: Build failed - dist directory not created"
    exit 1
fi

# List built files
echo "Built files:"
ls -la dist/

# Upload to PyPI
echo "Uploading to PyPI..."
twine upload dist/*

echo "Upload completed successfully!"
echo "Package uploaded to: https://pypi.org/project/clap-llm/"