@echo off
setlocal
cd /d "%~dp0"

echo === 1/6: download data ===
if exist "data\raw\qws" (
    echo Data already downloaded. Skipping step 1.
) else (
    uv run python scripts\01_download_data.py
    if errorlevel 1 goto error
)

echo.
echo === 2/6: prepare data ===
if exist "data\processed" (
    echo Data already prepared. Skipping step 2.
) else (
    uv run python scripts\02_prepare_data.py
    if errorlevel 1 goto error
)

echo.
echo === 3/6: validate ===
uv run python scripts\03_validate_baselines.py
if errorlevel 1 goto error

echo.
echo === 4/6: stable experiment ===
set ARGS=%*
uv run python scripts\04_run_stable_experiment.py %ARGS%
if errorlevel 1 goto error

echo.
echo === 5/6: drift experiment ===
:: The drift experiment takes --n-trials instead of --n-pools
if defined ARGS (
    set DRIFT_ARGS=%ARGS:--n-pools=--n-trials%
) else (
    set DRIFT_ARGS=
)
uv run python scripts\05_run_drift_experiment.py %DRIFT_ARGS%
if errorlevel 1 goto error

echo.
echo === 6/7: report ===
uv run python scripts\06_generate_report.py
if errorlevel 1 goto error

echo.
echo === 7/7: presentation figures ===
uv run python scripts\10_generate_presentation_figures.py
if errorlevel 1 goto error

echo.
echo === Done ===
goto :EOF

:error
echo.
echo [ERROR] Pipeline failed at the last step.
exit /b 1
