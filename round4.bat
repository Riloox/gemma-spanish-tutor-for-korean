@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Spanish tutor - round 4
rem Lets PyTorch grow its memory blocks instead of fragmenting them (the Sep 30 run died with 1.2 GiB reserved but unusable)
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ============================================================
echo  Round 4: new 208-item test set (frozen first), base scored on it,
echo  self-distilled SFT v4 (rewrite-first), GRPO round 4 with the
echo  rewrite reward, test vs the base, significance report.
echo  About 9-11 hours (test set ~1, distill ~1.5, SFT ~1,
echo  filter ~3, GRPO ~2.5, evals ~1).
echo.
echo  Needs in Ollama: gemma4:26b, gemma4:e2b, gemma4-tutor-grpo3
echo.
echo  Before leaving:
echo   - keep the laptop PLUGGED IN
echo   - don't close the lid (lid-close sleep can't be blocked)
echo   - the screen may turn off; that's fine
echo.
echo  Safe to stop anytime with Ctrl+C and re-run: it resumes.
echo  To turn the PC off at the end:  round4.bat shutdown
echo  Log:     logs\round4.log
echo  Result:  outputs\grpo4\summary.md
echo ============================================================
echo.
if /i "%~1"=="shutdown" (
    uv run python -u scripts\round4.py --shutdown
) else (
    uv run python -u scripts\round4.py
)
echo.
echo Finished. Check logs\round4.log
if /i not "%~1"=="shutdown" pause
