@echo off
echo ========================================
echo RESEARCH PAPER SEEDED EXPERIMENTS
echo ========================================
echo.
echo 📄 This runs COMPLETE training pipeline for multiple seeds
echo 🎯 Purpose: Generate statistically robust results for publication
echo 🔬 Method: Full retraining for each seed (not just evaluation)
echo 📊 Output: Mean ± Standard Deviation metrics
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

REM Change to AI directory
cd /d "c:\Users\prave\Desktop\Research Paper\FL, XAI\work\CICD  project\AI"

echo.
echo 🎲 Running RESEARCH-LEVEL seeded experiments...
echo 📋 Seeds: 42, 123, 999, 2024, 777 (5 seeds for robust statistics)
echo 📁 Output: AI\model_artifacts\
echo ⏱️  This will take time - full training for each seed...
echo.

REM Run the proper seeded experiments script
python model_development\run_seeded_experiments.py

echo.
echo ========================================
echo ✅ RESEARCH EXPERIMENTS COMPLETED!
echo 📊 Check AI\model_artifacts\ for results:
echo    • seeded_experiments_detailed_*.json
echo    • seeded_experiments_summary_*.json
echo    • research_paper_table_*.csv (ready for paper)
echo ========================================
echo 📄 Use "Paper Format" column directly in your Springer/Scopus paper!
echo.
pause
