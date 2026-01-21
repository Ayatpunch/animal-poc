@echo off
REM Script to activate virtual environment on Windows (Command Prompt)

REM Get the directory where the script is located
set SCRIPT_DIR=%~dp0

REM Check if venv exists
if not exist "%SCRIPT_DIR%venv" (
    echo Error: Virtual environment not found at %SCRIPT_DIR%venv
    echo Please create a virtual environment first using: python -m venv venv
    exit /b 1
)

REM Activate the virtual environment
call "%SCRIPT_DIR%venv\Scripts\activate.bat"

REM Check if activation was successful
if defined VIRTUAL_ENV (
    echo Virtual environment activated successfully!
    echo Location: %VIRTUAL_ENV%
    python --version
) else (
    echo Failed to activate virtual environment
    exit /b 1
)
