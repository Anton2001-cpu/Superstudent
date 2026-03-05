@echo off
title SuperStudent
echo.

if not exist venv (
    echo  No setup found. Running setup first...
    call setup.bat
)

call venv\Scripts\activate

echo  Starting SuperStudent...
echo  Opening browser at http://localhost:5000
echo  Press Ctrl+C to stop.
echo.

start "" http://localhost:5000
python app.py

pause
