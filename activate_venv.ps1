# Script to activate virtual environment on Windows (PowerShell)

# Get the directory where the script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check if venv exists
if (-Not (Test-Path "$ScriptDir\venv")) {
    Write-Host "Error: Virtual environment not found at $ScriptDir\venv" -ForegroundColor Red
    Write-Host "Please create a virtual environment first using: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Activate the virtual environment
& "$ScriptDir\venv\Scripts\Activate.ps1"

# Check if activation was successful
if ($env:VIRTUAL_ENV) {
    Write-Host "[SUCCESS] Virtual environment activated successfully!" -ForegroundColor Green
    Write-Host "Location: $env:VIRTUAL_ENV" -ForegroundColor Cyan
    $pythonVersion = python --version
    Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan
} else {
    Write-Host "[ERROR] Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}
