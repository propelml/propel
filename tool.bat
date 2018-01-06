@echo off
setlocal
set tool=%~dp0\tools\%1.js
shift
node "%tool%" %*
