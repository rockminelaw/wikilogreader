@echo off
rem Train the teacher on the GPU. Run from the Anaconda Prompt after:
rem   conda activate wlr
rem Extra options are passed through, e.g.  run_teacher.bat --epochs 1
setlocal
set PYTHONUTF8=1
set HF_HOME=%~dp0..\..\hf-cache
cd /d "%~dp0.."
python -m training.train_teacher --exports ..\wiki-export --out ..\runs\teacher-v1 %*
endlocal
