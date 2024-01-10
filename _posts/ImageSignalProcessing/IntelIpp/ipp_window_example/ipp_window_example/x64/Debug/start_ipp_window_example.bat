@echo off
setlocal
set BIN_PATH=%~dp0bin
set PATH=%BIN_PATH%;%PATH%
start "" "%~dp0ipp_window_example.exe"
endlocal
