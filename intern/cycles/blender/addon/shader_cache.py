# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Configure disk caches before device contexts are created, without touching NVIDIA's shared DB."""

import os
from pathlib import Path
import tempfile

_external = {key: os.environ.get(key) for key in
             ('CYCLES_CACHE_PATH', 'CUDA_CACHE_PATH', 'OPTIX_CACHE_PATH')}
_configured = False


def default_directory():
    if os.name == 'nt':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    root = base / 'Blender Foundation'
    current = root / 'QCycles' / 'shader_cache'
    legacy = root / 'Blender_GZ' / 'shader_cache'
    # Keep a previously populated shader cache available across the rename.
    return legacy if not current.exists() and legacy.is_dir() else current


def _writable(directory):
    directory.mkdir(parents=True, exist_ok=True)
    fd, probe = tempfile.mkstemp(prefix='.write-test-', dir=directory)
    os.close(fd)
    os.unlink(probe)


def configure():
    global _configured
    if _configured:
        return
    import bpy
    addon = bpy.context.preferences.addons.get(__package__)
    custom = addon.preferences.shader_cache_directory if addon else ''
    requested = Path(bpy.path.abspath(custom)).expanduser() if custom else default_directory()
    candidates = [requested]
    if custom:
        candidates.append(default_directory())
    datafiles = bpy.utils.user_resource('DATAFILES')
    if datafiles:
        candidates.append(Path(datafiles) / 'shader_cache')
    for directory in dict.fromkeys(candidates):
        try:
            paths = {key: directory / sub for key, sub in (
                ('CYCLES_CACHE_PATH', 'cycles'), ('CUDA_CACHE_PATH', 'cuda'),
                ('OPTIX_CACHE_PATH', 'optix')) if not _external[key]}
            for path in paths.values():
                _writable(path)
        except OSError as error:
            print(f"Cycles: 着色器缓存目录不可写：{directory}（{error}），尝试备用目录。", flush=True)
            continue
        for key, path in paths.items():
            os.environ[key] = str(path)
        _configured = True
        return
    # Rendering still works without a disk cache. Don't alter an explicit environment override.
    if not _external['CUDA_CACHE_PATH']:
        os.environ['CUDA_CACHE_DISABLE'] = '1'
    if not _external['OPTIX_CACHE_PATH']:
        os.environ['OPTIX_CACHE_MAXSIZE'] = '0'
    print("Cycles: 无可写的着色器缓存目录，继续渲染；着色器可能需要重新编译。", flush=True)
    _configured = True
