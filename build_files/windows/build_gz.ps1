# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
param(
  [string]$BuildDirectory = (Join-Path $PSScriptRoot '..\..\build'),
  [ValidateRange(1, 64)][int]$Jobs = 6,
  [string]$CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4',
  [string]$OptixRoot = 'C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0',
  [switch]$CleanFirst
)
$ErrorActionPreference = 'Stop'
$sourceDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$buildDirectoryFull = [IO.Path]::GetFullPath($BuildDirectory)
$cudaDirectory = [IO.Path]::GetFullPath($CudaRoot).Replace('\', '/')
$optixDirectory = [IO.Path]::GetFullPath($OptixRoot).Replace('\', '/')
$installDirectory = $buildDirectoryFull.Replace('\', '/') + '/bin/${BUILD_TYPE}'
if ($sourceDirectory -eq $buildDirectoryFull) { throw 'Use a separate build directory.' }
$logDirectory = Join-Path $buildDirectoryFull 'logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$cmakeExe = (Get-Command cmake -CommandType Application).Source
function Invoke-CMakeLogged([string[]]$Arguments, [string]$LogName) {
  # Windows PowerShell 5.1 turns redirected native stderr into ErrorRecords.
  # CMake warnings on stderr are normal; the process exit code determines success.
  $ErrorActionPreference = 'Continue'
  $log = Join-Path $logDirectory $LogName
  & $cmakeExe @Arguments *> $log
  if ($LASTEXITCODE -ne 0) { throw "CMake exited with code $LASTEXITCODE. See $log" }
}
$cmakeArgs = @(
  '-S', $sourceDirectory, '-B', $buildDirectoryFull,
  '-G', 'Visual Studio 17 2022', '-A', 'x64',
  "-DCMAKE_INSTALL_PREFIX:PATH=$installDirectory",
  '-DWITH_CYCLES=ON',
  '-DWITH_CYCLES_DEVICE_HIP=OFF', '-DWITH_CYCLES_DEVICE_HIPRT=OFF',
  '-DWITH_CYCLES_DEVICE_ONEAPI=OFF', '-DWITH_CYCLES_HIP_BINARIES=OFF',
  '-DWITH_CYCLES_OSL=ON', '-DWITH_CYCLES_PATH_GUIDING=ON',
  '-DWITH_OPENVDB=ON', '-DWITH_NANOVDB=ON', '-DWITH_INTERNATIONAL=ON',
  '-DWITH_UNITY_BUILD=ON', '-DWITH_GTESTS=OFF',
  "-DCUDA_TOOLKIT_ROOT_DIR:PATH=$cudaDirectory", "-DOPTIX_ROOT_DIR:PATH=$optixDirectory",
  '-C', (Join-Path $sourceDirectory 'build_files/cmake/config/blender_gz_windows.cmake')
)
Invoke-CMakeLogged $cmakeArgs 'configure.log'
$buildArgs = @('--build', $buildDirectoryFull, '--config', 'Release', '--target', 'INSTALL',
               '--parallel', $Jobs)
if ($CleanFirst) { $buildArgs += '--clean-first' }
Invoke-CMakeLogged $buildArgs 'build.log'
Write-Output (Join-Path $buildDirectoryFull 'bin\Release\blender.exe')
