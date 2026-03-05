@echo off
title SuperStudent — Setup
echo.
echo  Setting up SuperStudent...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  Python is not installed or not in your PATH.
    echo  Download it from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

if not exist venv (
    echo  Creating virtual environment...
    python -m venv venv
)

echo  Activating environment...
call venv\Scripts\activate

echo  Installing packages (this takes a minute the first time)...
pip install -r requirements.txt --quiet

echo.
echo  Setup complete!
echo  Now open the .env file and paste your OpenAI API key,
echo  then double-click start.bat to launch SuperStudent.
echo.
pause
