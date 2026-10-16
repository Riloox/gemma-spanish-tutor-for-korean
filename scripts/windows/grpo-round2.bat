@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
title Spanish tutor - GRPO round 2
rem Lets PyTorch grow its memory blocks instead of fragmenting them (the Sep 30 run died with 1.2 GiB reserved but unusable)
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ============================================================
echo  GRPO round 2: filter hard prompts, train, pick checkpoint,
echo  test on the held-out set, significance report.
echo  About 8 hours in total (training is ~6 of them).
echo.
echo  Before leaving:
echo   - keep the laptop PLUGGED IN
echo   - don't close the lid (lid-close sleep can't be blocked)
echo   - the screen may turn off; that's fine
echo.
echo  Safe to stop anytime with Ctrl+C and re-run: it resumes.
echo  To turn the PC off at the end:  scripts\windows\grpo-round2.bat shutdown
echo  Log:     logs\grpo-round2.log
echo  Result:  outputs\grpo2\summary.md
echo ============================================================
echo.
if /i "%~1"=="shutdown" (
    uv run python -u scripts\grpo_round2.py --shutdown
) else (
    uv run python -u scripts\grpo_round2.py
)
echo.
echo Finished. Check logs\grpo-round2.log
if /i not "%~1"=="shutdown" pause
