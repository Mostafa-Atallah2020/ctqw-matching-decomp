@echo off
REM ============================================================
REM CX Scaling: Erdos-Renyi graphs - Compression-Aware Heuristic
REM ============================================================
REM Runs cx_count_erdos_renyi_compute.py for each probability
REM separately, so each p gets its own output folder.
REM
REM Usage:  run_cx_scaling_erdos_renyi.bat
REM ============================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set GRAPH_DIR=%SCRIPT_DIR%graphs\erdos_renyi
set COMPUTE_SCRIPT=%SCRIPT_DIR%cx_count_erdos_renyi_compute.py

REM Probabilities to process
set PROBS=p0_01 p0_02 p0_03 p0_04 p0_05 p0_10 p0_20

echo.
echo ==============================================================
echo  COMPRESSION-AWARE HEURISTIC - Erdos-Renyi CX Scaling
echo ==============================================================

for %%P in (%PROBS%) do (
    echo.
    echo --------------------------------------------------------------
    echo  Processing %%P [compression_aware]
    echo --------------------------------------------------------------

    set OUTPUT_DIR=%SCRIPT_DIR%outputs_comp_aware\erdos_renyi\%%P

    python "%COMPUTE_SCRIPT%" ^
        "%GRAPH_DIR%\*_%%P_*.g6" ^
        -o "!OUTPUT_DIR!" ^
        --heuristic compression_aware

    if errorlevel 1 (
        echo [ERROR] Failed on %%P
    ) else (
        echo [OK] Completed %%P
    )
)

echo.
echo ==============================================================
echo  ALL DONE
echo ==============================================================
echo.
echo Results saved to: %SCRIPT_DIR%outputs_comp_aware\erdos_renyi\
echo.
echo To plot:
echo   python %SCRIPT_DIR%cx_count_erdos_renyi_plot.py %SCRIPT_DIR%outputs_comp_aware\erdos_renyi
echo.

endlocal
