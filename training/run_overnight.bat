@echo off
rem Two teacher runs on the v2 split, one after the other:
rem   teacher-v2-256  the same recipe as teacher-v1 (256 tokens)
rem   teacher-v2-384  reads up to 384 tokens, to test whether long edits suffer from the cut
rem Both score the training set too, which phase 3's distillation needs.
rem Run from a prompt where the venv is active.
setlocal
set PYTHONUTF8=1
set HF_HOME=%~dp0..\..\hf-cache
cd /d "%~dp0.."
python -m training.train_teacher --split v2 --epochs 1 --score-train --max-len 256 --exports ..\wiki-export --out ..\runs\teacher-v2-256
python -m training.train_teacher --split v2 --epochs 1 --score-train --max-len 384 --exports ..\wiki-export --out ..\runs\teacher-v2-384
endlocal
