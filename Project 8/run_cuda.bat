@echo off
rem Project 8 - CUDA GPU runs
setlocal
cd /d "%~dp0"
echo ==================================================================
echo  Project 8 - Spatial filtering on the GPU (CUDA)
echo  Note: run on my gaming desktop (AMD Ryzen 7 9700X 8-core,
echo        NVIDIA GeForce RTX 5070 Ti, Windows 11 Home)
echo ==================================================================
echo.
echo ##### 1) sample input (verified against the sequential filter)
bin\SpatialFilterCUDA.exe image256_edge.txt 1 images\cuda_out_edge.bmp
echo.
echo ##### 2) 8192 x 8192 generated image, 1 pass (verified)
bin\SpatialFilterCUDA.exe -gen 8192 8192 smooth 1
echo.
echo ##### 3) 8192 x 8192, 100 passes (checksum must equal the MPI 100-pass run)
bin\SpatialFilterCUDA.exe -gen 8192 8192 smooth 100
