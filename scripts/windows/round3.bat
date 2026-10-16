@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
title Spanish tutor - round 3
rem Lets PyTorch grow its memory blocks instead of fragmenting them (the Sep 30 run died with 1.2 GiB reserved but unusable)
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ============================================================
echo  Round 3: teacher items, SFT v3 (check-first), GRPO round 3,
echo  test on the held-out set vs the base, significance report.
echo  About 10-11 hours (teacher ~5, SFT ~1.5, GRPO ~2).
echo.
echo  Before leaving:
echo   - keep the laptop PLUGGED IN
echo   - don't close the lid (lid-close sleep can't be blocked)
echo   - the screen may turn off; that's fine
echo.
echo  Safe to stop anytime with Ctrl+C and re-run: it resumes.
echo  To turn the PC off at the end:  scripts\windows\round3.bat shutdown
echo  Log:     logs\round3.log
echo  Result:  outputs\grpo3\summary.md
echo ============================================================
echo.
if /i "%~1"=="shutdown" (
    uv run python -u scripts\round3.py --shutdown
) else (
    uv run python -u scripts\round3.py
)
echo.
echo Finished. Check logs\round3.log
if /i not "%~1"=="shutdown" pause
