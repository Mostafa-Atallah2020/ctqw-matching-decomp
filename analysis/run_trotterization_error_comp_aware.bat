@echo off
REM ============================================================
REM Trotterization 2-Norm Error: Greedy vs Compression-Aware
REM ============================================================
REM Runs trotterization_error_compute.py 5 times for each
REM heuristic (greedy and compression_aware), then generates
REM combined plots with mean +/- std bands.
REM
REM All outputs under analysis/error_plots/:
REM   data/greedy_{N}v/<timestamps>/raw_results.csv
REM   data/ca_{N}v/<timestamps>/raw_results.csv
REM   trotterization_error_{N}v.pdf
REM
REM Usage:  run_trotterization_error_comp_aware.bat
REM ============================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set GRAPH_DIR=%SCRIPT_DIR%graphs\connected
set COMPUTE_SCRIPT=%SCRIPT_DIR%trotterization_error_compute.py
set PLOT_SCRIPT=%SCRIPT_DIR%plot_trotterization_combined.py
set ROOT_DIR=%SCRIPT_DIR%error_plots
set DATA_DIR=%ROOT_DIR%\data

REM Trotter step range
set MIN_STEPS=1
set MAX_STEPS=20
set STEP_INC=2

REM Time values
set TIME_VALUES=0.1 0.5 1.0

REM Number of runs for variance estimation
set N_RUNS=5

REM Vertex counts and their graph files
set VERTICES=8 16 32
set GRAPHS_8=80graph_counting_8v.g6
set GRAPHS_16=200graph_counting_16v.g6
set GRAPHS_32=200graph_counting_32v.g6

echo.
echo ==============================================================
echo  TROTTERIZATION 2-NORM ERROR: GREEDY vs COMPRESSION-AWARE
echo  Trotter steps %MIN_STEPS% to %MAX_STEPS% (inc %STEP_INC%)
echo  Time values: %TIME_VALUES%
echo  %N_RUNS% runs per heuristic per vertex count
echo  Output root: %ROOT_DIR%
echo ==============================================================

for %%V in (%VERTICES%) do (
    echo.
    echo ==============================================================
    echo  Processing %%V vertices
    echo ==============================================================

    set GRAPH_FILE=%GRAPH_DIR%\!GRAPHS_%%V!
    set GREEDY_OUT=%DATA_DIR%\greedy_%%Vv
    set CA_OUT=%DATA_DIR%\ca_%%Vv

    REM ---- Greedy: 5 runs ----
    for /L %%R in (1,1,%N_RUNS%) do (
        echo.
        echo --- Greedy run %%R/%N_RUNS% for %%V vertices ---
        python "%COMPUTE_SCRIPT%" ^
            "!GRAPH_FILE!" ^
            -m %MIN_STEPS% -M %MAX_STEPS% -s %STEP_INC% ^
            -t %TIME_VALUES% ^
            --heuristic greedy ^
            -o "!GREEDY_OUT!"
    )

    REM ---- Compression-aware: 5 runs ----
    for /L %%R in (1,1,%N_RUNS%) do (
        echo.
        echo --- Comp-aware run %%R/%N_RUNS% for %%V vertices ---
        python "%COMPUTE_SCRIPT%" ^
            "!GRAPH_FILE!" ^
            -m %MIN_STEPS% -M %MAX_STEPS% -s %STEP_INC% ^
            -t %TIME_VALUES% ^
            --heuristic compression_aware ^
            -o "!CA_OUT!"
    )
)

echo.
echo ==============================================================
echo  GENERATING COMBINED PLOTS
echo ==============================================================

for %%V in (%VERTICES%) do (
    set GREEDY_DIR=%DATA_DIR%\greedy_%%Vv
    set CA_DIR=%DATA_DIR%\ca_%%Vv
    set OUT_PDF=%ROOT_DIR%\trotterization_error_%%Vv.pdf

    if exist "!GREEDY_DIR!" if exist "!CA_DIR!" (
        echo.
        echo --- Combined plot for %%V vertices ---
        python "%PLOT_SCRIPT%" ^
            --greedy "!GREEDY_DIR!" ^
            --comp-aware "!CA_DIR!" ^
            --output "!OUT_PDF!" ^
            --times 0.1 0.5 1.0 ^
            --title "N = %%V"
    ) else (
        echo [WARN] Missing data for %%V vertices
    )
)

echo.
echo ==============================================================
echo  ALL DONE
echo ==============================================================
echo.
echo All outputs under:
echo   %ROOT_DIR%\
echo     data\greedy_{N}v\<timestamps>\
echo     data\ca_{N}v\<timestamps>\
echo     trotterization_error_{N}v.pdf
echo.

endlocal
