@echo off
rem Builds the native versions (real MPI on all CPU cores + CUDA GPU).
rem Needs: Visual Studio 2022 Build Tools (C++), MS-MPI + MS-MPI SDK, CUDA Toolkit.
rem The C* simulator program needs no build - it is compiled by cstar.exe.
setlocal
cd /d "%~dp0"
if not defined MSMPI_INC set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include\"
if not defined MSMPI_LIB64 set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\"
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
rem strip trailing backslashes (they would escape the closing quote)
if "%MSMPI_INC:~-1%"=="\" set "MSMPI_INC=%MSMPI_INC:~0,-1%"
if "%MSMPI_LIB64:~-1%"=="\" set "MSMPI_LIB64=%MSMPI_LIB64:~0,-1%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || goto :fail
if not exist bin mkdir bin

echo [1/3] imagetool
cl /nologo /O2 /W3 /D_CRT_SECURE_NO_WARNINGS native\imagetool.c /Fobin\ /Fe:bin\imagetool.exe || goto :fail

echo [2/3] SpatialFilterMPI (MS-MPI)
cl /nologo /O2 /arch:AVX2 /W3 /D_CRT_SECURE_NO_WARNINGS /I"%MSMPI_INC%" native\SpatialFilterMPI.c ^
   /Fobin\ /Fe:bin\SpatialFilterMPI.exe /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib || goto :fail

echo [3/3] SpatialFilterCUDA
"%CUDA_PATH%\bin\nvcc.exe" -O3 -arch=native -D_CRT_SECURE_NO_WARNINGS -o bin\SpatialFilterCUDA.exe native\SpatialFilterCUDA.cu || goto :fail

echo Build OK
exit /b 0
:fail
echo BUILD FAILED
exit /b 1
