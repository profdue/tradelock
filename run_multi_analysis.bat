@echo off
echo ========================================
echo MULTI-PAIR STATE ANALYZER
echo ========================================
echo.

cd /d "C:\Trading_Analysis"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Install from: python.org (check "Add to PATH")
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
python -m pip install pandas --quiet

echo.
echo Checking for data files...

set FILES_EXIST=0
if exist "Gold_StateLog.csv" (
    echo ✓ GOLD# data found
    set /a FILES_EXIST+=1
)
if exist "Eurusd_StateLog.csv" (
    echo ✓ EURUSD data found
    set /a FILES_EXIST+=1
)
if exist "Gbpusd_StateLog.csv" (
    echo ✓ GBPUSD data found
    set /a FILES_EXIST+=1
)

if %FILES_EXIST% == 0 (
    echo.
    echo ❌ No data files found!
    echo Copy from MT5: MQL5\Files\*_StateLog.csv
    pause
    exit /b 1
)

echo.
echo Running multi-pair analysis...
python multi_pair_analyzer.py

echo.
if exist "persistent_states_summary.csv" (
    echo Opening results in Excel...
    timeout /t 2 >nul
    start excel.exe "persistent_states_summary.csv"
)

echo.
pause
