@echo off
setlocal

:: Get a safe timestamp for folder naming
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set mytime=%mytime: =%
set TIMESTAMP=%mydate%_%mytime%

set BACKUP_DIR=results\past_results_%TIMESTAMP%

echo ========================================================
echo Archiving previous results to prevent progress loss...
echo ========================================================
mkdir %BACKUP_DIR%\tables 2>nul
mkdir %BACKUP_DIR%\figures 2>nul

move results\tables\stable_results.csv %BACKUP_DIR%\tables\ 2>nul
move results\tables\drift_results.csv %BACKUP_DIR%\tables\ 2>nul
move results\figures\*.* %BACKUP_DIR%\figures\ 2>nul

echo Old results and figures moved safely to: %BACKUP_DIR%
echo.

echo ========================================================
echo Running a Fresh Smoke Test (1 pool per task)...
echo ========================================================
:: Since n-pools is 1, it will only do 1 pool per task. It will still take some time
:: due to CPU inference, but you can see the very first rows being written in real-time.
uv run python scripts\04_run_stable_experiment.py --n-pools 1 --yes

echo ========================================================
echo Running Drift Experiment (1 pool per task)...
echo ========================================================
uv run python scripts\05_run_drift_experiment.py --n-trials 1 --yes

echo ========================================================
echo Generating Reports and Figures...
echo ========================================================
uv run python scripts\06_generate_report.py
uv run python scripts\08_generate_extended_analysis.py
uv run python scripts\10_generate_presentation_figures.py

echo Done!
