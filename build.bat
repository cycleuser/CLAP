@echo off
REM Build script for clap-llm package
REM This script builds the Python package for distribution

echo Building clap-llm package...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9 or higher.
    exit /b 1
)

REM Clean previous builds
if exist dist (
    echo Cleaning previous builds...
    rmdir /s /q dist
)

if exist build (
    rmdir /s /q build
)

if exist *.egg-info (
    rmdir /s /q *.egg-info
)

REM Build the package
echo Building source distribution and wheel...
python -m build

if errorlevel 1 (
    echo Error: Build failed.
    exit /b 1
)

echo Build completed successfully!
echo Files in dist/ directory:
dir dist\*

pause