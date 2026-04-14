@echo off
REM ============================================================
REM Connected Graphs: Compression-Aware Matching vs Pauli
REM ============================================================
REM Runs ctqw_matching_vs_pauli.py 5 times per vertex count
REM with --heuristic compression_aware, then aggregates with
REM parse_run_logs.py.
REM
REM Output: outputs_comp_aware/connected/connected_{N}v/
REM
REM Usage:  run_connected_comp_aware.bat
REM ============================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set GRAPH_DIR=%SCRIPT_DIR%graphs\connected
set COMPUTE_SCRIPT=%SCRIPT_DIR%ctqw_matching_vs_pauli.py
set PARSE_SCRIPT=%SCRIPT_DIR%parse_run_logs.py

REM Number of runs per vertex count (for transpiler variance)
set N_RUNS=5

REM Vertex counts and their graph files
set VERTICES=8 16 32 64 128
set GRAPHS_8=80graph_counting_8v.g6
set GRAPHS_16=200graph_counting_16v.g6
set GRAPHS_32=200graph_counting_32v.g6
set GRAPHS_64=200graph_counting_64v.g6
set GRAPHS_128=200graph_counting_128v.g6

echo.
echo ==============================================================
echo  COMPRESSION-AWARE MATCHING vs PAULI - Connected Graphs
echo  %N_RUNS% runs per vertex count
echo ==============================================================

for %%V in (%VERTICES%) do (
    echo.
    echo ==============================================================
    echo  Processing %%V vertices
    echo ==============================================================

    set GRAPH_FILE=%GRAPH_DIR%\!GRAPHS_%%V!
    set OUTPUT_SUB=connected_%%Vv

    for /L %%R in (1,1,%N_RUNS%) do (
        echo.
        echo --------------------------------------------------------------
        echo  Run %%R/%N_RUNS% for %%V vertices [compression_aware]
        echo --------------------------------------------------------------

        python "%COMPUTE_SCRIPT%" ^
            "!GRAPH_FILE!" ^
            -o "!OUTPUT_SUB!" ^
            --heuristic compression_aware

        if errorlevel 1 (
            echo [ERROR] Failed on %%V vertices, run %%R
        ) else (
            echo [OK] Completed %%V vertices, run %%R
        )
    )

    echo.
    echo --------------------------------------------------------------
    echo  Aggregating results for %%V vertices
    echo --------------------------------------------------------------

    REM Find the output folder with timestamp subfolders
    REM Structure: outputs_comp_aware/matching_vs_pauli/connected_{N}v/counting_{N}v/
    set AGG_DIR=%SCRIPT_DIR%outputs_comp_aware\matching_vs_pauli\!OUTPUT_SUB!\counting_%%Vv

    if exist "!AGG_DIR!" (
        python "%PARSE_SCRIPT%" "!AGG_DIR!"

        if errorlevel 1 (
            echo [ERROR] Aggregation failed for %%V vertices
        ) else (
            echo [OK] Aggregated %%V vertices
        )
    ) else (
        echo [WARN] Output folder not found: !AGG_DIR!
    )
)

echo.
echo ==============================================================
echo  ALL DONE
echo ==============================================================
echo.
echo Results saved to:
echo   %SCRIPT_DIR%outputs_comp_aware\matching_vs_pauli\
echo.
echo To aggregate manually:
echo   python %PARSE_SCRIPT% outputs_comp_aware\matching_vs_pauli\connected_16v\counting_16v
echo.

endlocal
