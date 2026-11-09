@echo off
rem Japanese text stays out of this file: cmd reads it in the console code page.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
