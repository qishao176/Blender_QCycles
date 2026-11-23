# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

file(STRINGS "${MANIFEST}" kernels)
set(info "")
foreach(kernel IN LISTS kernels)
  file(TIMESTAMP "${KERNEL_DIR}/${kernel}" compiled_at "%Y-%m-%dT%H:%M:%SZ" UTC)
  file(MD5 "${KERNEL_DIR}/${kernel}" digest)
  string(APPEND info "${kernel} ${digest} ${compiled_at}\n")
endforeach()
file(WRITE "${OUTPUT}" "${info}")
