@echo off
setlocal EnableExtensions DisableDelayedExpansion
if exist "%~dp0blender.exe" goto prewarm
echo QCycles: blender.exe was not found next to "%~f0". 1>&2
exit /b 1

:prewarm
echo Prewarming the built-in Cycles kernels and denoisers for all available OptiX GPUs...
"%~dp0blender.exe" --background --disable-autoexec --addons cycles --log cycles --log-level info --python-exit-code 1 --python-expr "import bpy, sys; result = bpy.ops.cycles.prewarm_optix_cache(); 'FINISHED' in result or sys.exit(1)"
exit /b %errorlevel%
