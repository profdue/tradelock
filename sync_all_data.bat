@echo off
REM Copy all CSV files from MT5 to analysis folder
xcopy "C:\Users\davidug.SIFAXGROUP\AppData\Roaming\MetaQuotes\Terminal\BA78EA1631820D7AEF052166A20E7B1A\MQL5\Files\*_StateLog.csv" "C:\Trading_Analysis\" /Y
echo Data copied at %time%
pause
