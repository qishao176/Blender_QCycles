# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Supplement the existing QCycles build without replacing its device or install settings.
# DLSS is loaded from the NVIDIA driver at runtime; only SDK headers are needed here.
if(NOT DLSS_SDK_ROOT)
  if(DEFINED ENV{DLSS_SDK_ROOT} AND NOT "$ENV{DLSS_SDK_ROOT}" STREQUAL "")
    set(DLSS_SDK_ROOT "$ENV{DLSS_SDK_ROOT}")
  else()
    get_filename_component(_gz_source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    set(DLSS_SDK_ROOT "${_gz_source_root}/lib/dlss")
    unset(_gz_source_root)
  endif()
endif()
file(TO_CMAKE_PATH "${DLSS_SDK_ROOT}" DLSS_SDK_ROOT)
set(DLSS_SDK_ROOT "${DLSS_SDK_ROOT}" CACHE PATH "NVIDIA DLSS SDK root" FORCE)

foreach(_gz_dlss_header nvsdk_ngx.h nvsdk_ngx_defs_dlssd.h nvsdk_ngx_loader.h)
  if(NOT EXISTS "${DLSS_SDK_ROOT}/include/${_gz_dlss_header}")
    message(FATAL_ERROR
      "QCycles requires the DLSS SDK headers (v310.9.1 or compatible). "
      "Missing: ${DLSS_SDK_ROOT}/include/${_gz_dlss_header}. "
      "Install the NVIDIA/DLSS SDK in lib/dlss or set DLSS_SDK_ROOT.")
  endif()
endforeach()
unset(_gz_dlss_header)

set(DLSS_INCLUDE_DIR "${DLSS_SDK_ROOT}/include" CACHE PATH "DLSS SDK headers" FORCE)
set(WITH_DLSS ON CACHE BOOL "Enable NVIDIA DLSS Ray Reconstruction support" FORCE)
message(STATUS "QCycles: DLSS Ray Reconstruction enabled; SDK: ${DLSS_SDK_ROOT}")
