@echo off
rem Phase 3: distil the teacher into small students. Each run is skipped if
rem its metrics.json already exists, so this can be started again after an
rem interruption. Add "shutdown" to turn the PC off two minutes after the
rem last run:   code\training\run_students.bat shutdown
rem The "-hard" runs use the true labels only (no teacher), for comparison.
setlocal
set PYTHONUTF8=1
set HF_HOME=%~dp0..\..\hf-cache
cd /d "%~dp0.."
if not exist ..\runs\student-tiny\metrics.json python -m training.distil --teacher ..\runs\teacher-v2-256 --exports ..\wiki-export --epochs 3 --model google/bert_uncased_L-2_H-128_A-2 --lr 3e-4 --alpha 0.3 --out ..\runs\student-tiny
if not exist ..\runs\student-tiny-hard\metrics.json python -m training.distil --teacher ..\runs\teacher-v2-256 --exports ..\wiki-export --epochs 3 --model google/bert_uncased_L-2_H-128_A-2 --lr 3e-4 --alpha 1 --out ..\runs\student-tiny-hard
if not exist ..\runs\student-mini\metrics.json python -m training.distil --teacher ..\runs\teacher-v2-256 --exports ..\wiki-export --epochs 3 --model google/bert_uncased_L-4_H-256_A-4 --lr 1e-4 --alpha 0.3 --out ..\runs\student-mini
if not exist ..\runs\student-mini-hard\metrics.json python -m training.distil --teacher ..\runs\teacher-v2-256 --exports ..\wiki-export --epochs 3 --model google/bert_uncased_L-4_H-256_A-4 --lr 1e-4 --alpha 1 --out ..\runs\student-mini-hard
if not exist ..\runs\student-electra\metrics.json python -m training.distil --teacher ..\runs\teacher-v2-256 --exports ..\wiki-export --epochs 3 --model google/electra-small-discriminator --lr 2e-4 --alpha 0.3 --out ..\runs\student-electra
if /i "%~1"=="shutdown" shutdown /s /t 120
endlocal
