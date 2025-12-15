@echo off
REM Get the directory where this script is located
SET SCRIPT_DIR=%~dp0

REM Activate the virtual environment
call "%SCRIPT_DIR%venv\Scripts\activate.bat"

REM Run the start_system.py script
python "%SCRIPT_DIR%start_system.py" %*