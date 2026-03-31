@echo off
title Memory Monitor
echo ============================================
echo  Memory Monitor (5s interval)
echo  Close this window to stop
echo ============================================
echo.
:loop
for /f "skip=1" %%a in ('wmic OS get FreePhysicalMemory ^| findstr [0-9]') do (
    set /a mb=%%a/1024
)
for /f "tokens=2 delims==" %%a in ('wmic cpu get LoadPercentage /value ^| findstr [0-9]') do (
    set cpu=%%a
)
echo [%time:~0,8%] Avail: %mb% MB ^| CPU: %cpu%%%
timeout /t 5 /nobreak >nul
goto loop
