@echo off
:: W4A Environment Setup Script
:: 
:: Sets up the W4A reinforcement learning environment
:: 
:: Requirements:
:: - Python 3.9 (required for SimulationInterface compatibility)
:: - Virtual environment recommended
::
:: Usage: setup.bat
:: Note: This assumes 'python3' is used to call python 3.9 (as expected by pyenv-win). This might need to change to 'python' if python 3.9 is installed natively.

setlocal

echo Setting up W4A Environment...

:: Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
set PYTHON_MAJOR_MINOR=%PYTHON_VERSION:~0,3%

if not "%PYTHON_MAJOR_MINOR%"=="3.9" (
    echo WARNING: Python 3.9 recommended for SimulationInterface compatibility
    echo Current version: %PYTHON_VERSION%
    echo Continuing anyway...
    echo.
)

echo Installing SimulationInterface...
:: Note: Cannot use -e due to compiled extensions
call python3 -m pip install ./SimulationInterface/

echo.
echo Installing w4a dependencies...
call python3 -m pip install -r requirements.txt

echo.
echo Installing w4a in development mode...
call python3 -m pip install -e .

echo.
echo Running tests...
call python3 -m pytest tests/test_basic_multiagent.py -v

echo.
echo.
echo Setup complete
echo.
echo Installed:
echo   - SimulationInterface (with compiled binaries)
echo   - w4a and dependencies
echo.

endlocal
pause