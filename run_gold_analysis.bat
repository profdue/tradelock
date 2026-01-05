@echo off
echo ========================================
echo GOLD# State Analyzer
echo ========================================
echo.

cd /d "C:\Trading_Analysis"

REM Check if file exists
if not exist "StateLog.csv" (
    echo ERROR: StateLog.csv not found!
    echo.
    echo Copy from MT5: 
    echo MQL5\Files\StateLog.csv
    pause
    exit /b 1
)

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found!
    echo Please install Python from: python.org
    pause
    exit /b 1
)

echo Python found. Running analysis...
python analyze_states.py

echo.
pause
