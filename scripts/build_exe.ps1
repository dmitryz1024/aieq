$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$venvPython = ".\.venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython -c "import PyInstaller, llama_cpp" 2>$null
    if ($LASTEXITCODE -eq 0) {
        & $venvPython -m PyInstaller --noconfirm AIEQ.spec
    } else {
        uv run --extra build --extra ai pyinstaller --noconfirm AIEQ.spec
    }
} else {
    uv run --extra build --extra ai pyinstaller --noconfirm AIEQ.spec
}

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "Built: dist\AIEQ\AIEQ.exe"
