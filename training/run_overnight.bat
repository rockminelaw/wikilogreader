@echo off
rem Overnight teacher runs on the v2 split. A run whose metrics.json already
rem exists is skipped, so this can be started again after an interruption.
rem Add "shutdown" to turn the PC off two minutes after the last run:
rem     code\training\run_overnight.bat shutdown
setlocal
set PYTHONUTF8=1
set HF_HOME=%~dp0..\..\hf-cache
cd /d "%~dp0.."
if not exist ..\runs\teacher-v2-256\metrics.json python -m training.train_teacher --split v2 --epochs 1 --score-train --max-len 256 --exports ..\wiki-export --out ..\runs\teacher-v2-256
if not exist ..\runs\teacher-v2-384\metrics.json python -m training.train_teacher --split v2 --epochs 1 --score-train --max-len 384 --exports ..\wiki-export --out ..\runs\teacher-v2-384
if /i "%~1"=="shutdown" shutdown /s /t 120
endlocal
