@echo off
rem Project 8 - run the hypercube spatial filter in Bruce Lester's C* simulator
rem (128 simulated processors = 7-dimensional hypercube) and the sequential
rem C* baseline, for one input file.   usage: run_simulator.bat [filter]
rem    filter = smooth (default) | edge | sharpen
rem
rem Same thing by hand inside cstar.exe:
rem    MPI ON
rem    INPUT = image256_smooth.txt
rem    OUTPUT = out_smooth.pgm
rem    OPEN SpatialFilterHypercube.c
rem    RUN
setlocal
cd /d "%~dp0"
set F=%1
if "%F%"=="" set F=smooth
set "CMDS=%TEMP%\cstar_cmds.txt"
set "OUT=%TEMP%\cstar_out.txt"
echo ==================================================================
echo  Project 8 - Spatial filtering on a 7-dimensional hypercube
echo  C* simulator (Bruce P. Lester), MPI mode, ARCHITECTURE HYPERCUBE(7)
echo  Note: run on my gaming desktop (AMD Ryzen 7 9700X 8-core,
echo        NVIDIA GeForce RTX 5070 Ti, Windows 11 Home)
echo ==================================================================
echo.
echo ---------- PARALLEL: SpatialFilterHypercube.c, input image256_%F%.txt
> "%CMDS%" (
  echo mpi on
  echo input = image256_%F%.txt
  echo output = out_%F%.pgm
  echo open SpatialFilterHypercube.c
  echo run
  echo utilization 0:7
  echo exit
)
.\cstar.exe < "%CMDS%" > "%OUT%"
findstr /v /r /c:"^\*$" /c:"^ *$" "%OUT%"
echo.
echo ---------- program performance summary (written by node 0 to out_%F%.pgm)
findstr /b "#" out_%F%.pgm& echo.& echo ---------- simulator summary for the parallel run& findstr /c:"EXECUTION TIME" /c:"SPEEDUP" /c:"PROCESSORS USED" out_%F%.pgm
echo.
echo ---------- SEQUENTIAL: SpatialFilterSequential.c, same input, 1 processor
> "%CMDS%" (
  echo input = image256_%F%.txt
  echo output = out_seq_%F%.pgm
  echo open SpatialFilterSequential.c
  echo run
  echo exit
)
.\cstar.exe < "%CMDS%" > "%OUT%"
findstr /c:"EXECUTION TIME" /c:"SPEEDUP" /c:"PROCESSORS USED" out_seq_%F%.pgm
echo.
echo ---------- correctness check against an independent sequential filter
bin\imagetool.exe verify image256_%F%.txt out_%F%.pgm
bin\imagetool.exe verify image256_%F%.txt out_seq_%F%.pgm
if not exist images mkdir images
bin\imagetool.exe tobmp image256_%F%.txt images\input_%F%.bmp >nul
bin\imagetool.exe tobmp out_%F%.pgm images\sim_out_%F%.bmp >nul
echo (input and filtered images written to images\ as .bmp)
