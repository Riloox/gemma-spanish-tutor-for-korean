@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Spanish tutor - overnight data generation
echo ============================================================
echo  Step 2c overnight job: teacher exam + explanation data
echo.
echo  Before bed:
echo   - keep the laptop PLUGGED IN
echo   - don't close the lid (lid-close sleep can't be blocked)
echo   - the screen may turn off; that's fine
echo.
echo  Safe to stop anytime with Ctrl+C and re-run: it resumes.
echo  Log: logs\overnight-*.log
echo ============================================================
echo.
uv run python -u scripts\overnight.py
echo.
echo Finished. Check the log in the logs folder.
pause
