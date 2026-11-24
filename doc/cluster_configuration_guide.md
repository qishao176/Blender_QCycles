# Cluster Configuration Guide / 群集配置指南

## Prewarm a render node / 预热渲染节点

Run this PowerShell command from the folder containing the QCycles `blender.exe`, using the same OS account and cache settings as the render worker. One invocation warms every available OptiX GPU on the node in sequence. It runs in the background without opening a scene.

在 QCycles 的 `blender.exe` 所在目录打开 PowerShell，用渲染进程的系统账户和缓存设置运行下列命令。一次调用会依次预热该节点上所有可用的 OptiX GPU，整个过程在后台完成，无需打开场景。

```powershell
.\blender.exe -b --log cycles --log-level info --python-exit-code 1 --python-expr "import bpy; bpy.ops.cycles.prewarm_optix_cache()"
```

The log shows progress for each GPU. Exit code `0` indicates success. If interrupted, run the command again to reuse completed entries. The first run can take tens of minutes. Run it again after changing the QCycles kernels, GPU model or NVIDIA driver.

日志会显示每张 GPU 的进度，退出码 `0` 表示成功。中断后重新运行同一命令即可复用已写入的条目。首次预热可能需要数十分钟。更换 QCycles 内核、GPU 型号或 NVIDIA 驱动后，请重新运行。

## Cache location / 缓存位置

The command uses the Cycles shader cache directory configured in Preferences. To set a dedicated location on a Windows worker, run the following before launching Blender. Give the render service the same environment so that later jobs find both caches. Use a local directory on each worker.

默认使用偏好设置中配置的 Cycles 着色器缓存目录。如需在 Windows 节点上指定位置，可在启动 Blender 前执行下列命令。渲染服务也应使用相同的环境变量，以便后续任务同时命中 CUDA 和 OptiX 缓存。每个节点使用自己的本地目录。

```powershell
$cacheRoot = Join-Path $env:LOCALAPPDATA 'QCycles\worker-cache'
$env:CUDA_CACHE_PATH = Join-Path $cacheRoot 'cuda'
$env:OPTIX_CACHE_PATH = Join-Path $cacheRoot 'optix'
New-Item -ItemType Directory -Force -Path $env:CUDA_CACHE_PATH, $env:OPTIX_CACHE_PATH | Out-Null
```

## Scope and verification / 覆盖范围与验证

Prewarming covers the built-in render kernels and OptiX denoisers, including motion blur, curves, point clouds, photon caustics and VCM. Native material closures and repeat zones use the shared shader kernels. Custom OSL shaders and cameras compile from their scene data separately.

预热覆盖内置渲染内核及 OptiX 降噪器，包括运动模糊、曲线、点云、光子焦散和 VCM。原生材质的闭包与重复区共用着色内核。自定义 OSL 着色器和 OSL 相机仍需按具体场景单独编译。

To verify reuse, rerun the first command in a new process with `--log-level debug` and look for `Cache hit for key`. On one RTX 5070, the warm run took 20 seconds inside the operator and recorded 832 hits with zero misses. The two caches occupied about 705 MiB together. These figures depend on the build, GPU and driver.

要验证复用，在新的 Blender 进程中重新运行第一条命令，将日志级别改为 `--log-level debug`，查找 `Cache hit for key`。一次 RTX 5070 实测中，热缓存下接口耗时 20 秒，命中 832 次、未命中 0 次，两份缓存合计约 705 MiB。实际耗时和大小取决于构建、GPU 和驱动。
