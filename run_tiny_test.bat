@echo off
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmm"') do set TIMESTAMP=%%i
set BACKUP_DIR=results\past_results_%TIMESTAMP%

echo ========================================================
echo Archiving previous results to prevent progress loss...
echo ========================================================
mkdir %BACKUP_DIR%\tables 2>nul
mkdir %BACKUP_DIR%\figures 2>nul
mkdir %BACKUP_DIR%\report 2>nul

move results\tables\stable_results.csv %BACKUP_DIR%\tables\ 2>nul
move results\tables\drift_results.csv %BACKUP_DIR%\tables\ 2>nul
move results\figures\*.* %BACKUP_DIR%\figures\ 2>nul
move results\report\*.* %BACKUP_DIR%\report\ 2>nul

echo ========================================================
echo Running fresh TINY test...
echo ========================================================

powershell -NoProfile -Command "$start = Get-Date; uv run python scripts/04_run_stable_experiment.py --n-pools 1 --yes; if ($?) { Write-Host \"Step 04 elapsed: $((New-TimeSpan -Start $start -End (Get-Date)).TotalSeconds) seconds\" } else { exit 1 }"
if %errorlevel% neq 0 exit /b 1

powershell -NoProfile -Command "$start = Get-Date; uv run python scripts/05_run_drift_experiment.py --n-trials 1 --yes; if ($?) { Write-Host \"Step 05 elapsed: $((New-TimeSpan -Start $start -End (Get-Date)).TotalSeconds) seconds\" } else { exit 1 }"
if %errorlevel% neq 0 exit /b 1

powershell -NoProfile -Command "$start = Get-Date; uv run python scripts/11_run_model_comparison.py --models qwen2.5:1.5b --reasoners DirectWeight,Classification --n-pools 1; if ($?) { Write-Host \"Step 11 elapsed: $((New-TimeSpan -Start $start -End (Get-Date)).TotalSeconds) seconds\" } else { exit 1 }"
if %errorlevel% neq 0 exit /b 1

powershell -NoProfile -Command "$start = Get-Date; uv run python scripts/06_generate_report_figures.py; if ($?) { Write-Host \"Step 06 elapsed: $((New-TimeSpan -Start $start -End (Get-Date)).TotalSeconds) seconds\" } else { exit 1 }"
if %errorlevel% neq 0 exit /b 1

powershell -NoProfile -Command "$start = Get-Date; uv run python scripts/07_generate_report_tables.py; if ($?) { Write-Host \"Step 07 elapsed: $((New-TimeSpan -Start $start -End (Get-Date)).TotalSeconds) seconds\" } else { exit 1 }"
if %errorlevel% neq 0 exit /b 1

echo Finding latest compare_ directory...
for /f "delims=" %%D in ('powershell -NoProfile -Command "Get-ChildItem -Path results\tables -Directory -Filter compare_* | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Select-Object -ExpandProperty FullName"') do set LATEST_COMPARE=%%D

powershell -NoProfile -Command "$start = Get-Date; uv run python scripts/12_make_report.py --csv \"!LATEST_COMPARE!\model_comparison.csv\"; if ($?) { Write-Host \"Step 12 elapsed: $((New-TimeSpan -Start $start -End (Get-Date)).TotalSeconds) seconds\" } else { exit 1 }"
if %errorlevel% neq 0 exit /b 1

powershell -NoProfile -Command "$start = Get-Date; uv run python scripts/verify.py; if ($?) { Write-Host \"Verify elapsed: $((New-TimeSpan -Start $start -End (Get-Date)).TotalSeconds) seconds\" } else { exit 1 }"
if %errorlevel% neq 0 exit /b 1

echo ========================================================
echo Done! All steps finished successfully.
echo Output paths:
echo Figures: results\figures
echo Tables: results\tables
echo Report: results\report
echo ========================================================
