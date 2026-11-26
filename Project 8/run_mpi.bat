@echo off
rem Project 8 - real MPI (MS-MPI) runs on all 8 CPU cores.
rem usage: run_mpi.bat [sample | scaling | sustained | all]   (default all)
rem   sample    - sample input on the full 7-D hypercube (128 ranks), verified
rem   scaling   - 1/2/4/8 ranks on a large generated image
rem   sustained - long 8-rank run (watch Task Manager: all 8 cores busy)
setlocal
cd /d "%~dp0"
set "MPIEXEC=C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
set PART=%1
if "%PART%"=="" set PART=all
echo ==================================================================
echo  Project 8 - Spatial filtering, real MPI on the CPU (MS-MPI)
echo  Note: run on my gaming desktop (AMD Ryzen 7 9700X 8-core,
echo        NVIDIA GeForce RTX 5070 Ti, Windows 11 Home)
echo ==================================================================
if "%PART%"=="sample" goto sample
if "%PART%"=="scaling" goto scaling
if "%PART%"=="sustained" goto sustained

:sample
echo.
echo ##### sample input image256_smooth.txt, 128 ranks = 7-dimensional hypercube
"%MPIEXEC%" -n 128 bin\SpatialFilterMPI.exe image256_smooth.txt 1 images\mpi_out_smooth.bmp
echo.
echo ##### sample input image256_edge.txt, 8 ranks = 3-dimensional hypercube
"%MPIEXEC%" -n 8 bin\SpatialFilterMPI.exe image256_edge.txt 1 images\mpi_out_edge.bmp | findstr /c:"MPI ranks" /c:"verify" /c:"checksum" /c:"wrote"
if not "%PART%"=="all" goto :eof

:scaling
echo.
echo ##### scaling: 8192 x 8192 image, 100 passes, 1/2/4/8 ranks (one rank per core)
for %%n in (1 2 4 8) do (
  echo.
  echo ---------- %%n rank^(s^)
  "%MPIEXEC%" -n %%n bin\SpatialFilterMPI.exe -gen 8192 8192 smooth 100 | findstr /c:"MPI ranks" /c:"filter passes" /c:"throughput" /c:"balance" /c:"CPU USAGE" /c:"whole run" /c:"checksum"
)
if not "%PART%"=="all" goto :eof

:sustained
echo.
echo ##### sustained run on all 8 cores: 8192 x 8192, 1000 passes
"%MPIEXEC%" -n 8 bin\SpatialFilterMPI.exe -gen 8192 8192 smooth 1000
