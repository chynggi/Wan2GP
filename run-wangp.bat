@echo off
rem WanGP launcher (Windows) - delegates to run-wangp.ps1
where pwsh >nul 2>&1
if %ERRORLEVEL%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-wangp.ps1" %*
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-wangp.ps1" %*
)
exit /b %ERRORLEVEL%
