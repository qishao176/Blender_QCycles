# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Shared initial cache for the QCycles Windows x64 release. Existing explicit device
# choices survive reconfiguration. A fresh release contains the redistributable PTX.
include("${CMAKE_CURRENT_LIST_DIR}/blender_gz_dlss.cmake")
set(WITH_CYCLES_DEVICE_CUDA ON CACHE BOOL "Build the CUDA render device")
set(WITH_CYCLES_CUDA_BINARIES ON CACHE BOOL "Build CUDA kernels for distribution")
set(CYCLES_CUDA_BINARIES_ARCH compute_75 CACHE STRING "CUDA architectures")
set(WITH_CYCLES_DEVICE_OPTIX ON CACHE BOOL "Build the OptiX render device")

if(WITH_CYCLES_DEVICE_OPTIX AND NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
  # A copied cache can still point to an SDK that no longer exists on this host.
  unset(OPTIX_INCLUDE_DIR CACHE)
  if(OPTIX_ROOT_DIR)
    set(_gz_optix_roots "${OPTIX_ROOT_DIR}")
  elseif(DEFINED ENV{OPTIX_ROOT_DIR})
    set(_gz_optix_roots "$ENV{OPTIX_ROOT_DIR}")
  else()
    file(GLOB _gz_optix_roots "$ENV{ProgramData}/NVIDIA Corporation/OptiX SDK *")
    list(SORT _gz_optix_roots COMPARE NATURAL ORDER DESCENDING)
  endif()
  find_path(OPTIX_INCLUDE_DIR optix.h HINTS ${_gz_optix_roots} PATH_SUFFIXES include)
  unset(_gz_optix_roots)
  if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "QCycles OptiX SDK headers not found. Set OPTIX_ROOT_DIR to the SDK directory.")
  endif()
endif()
