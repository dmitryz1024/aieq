param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Test-PythonModule {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string]$Module
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $script = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$Module') else 1)"
        $ErrorActionPreference = "Continue"
        & $Python -c $script *> $null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string[]]$Command
    )

    $executable = $Command[0]
    $arguments = @($Command | Select-Object -Skip 1)
    & $executable @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$venvPython = ".\.venv\Scripts\python.exe"

if ($Clean) {
    Remove-Item -LiteralPath ".\build", ".\dist" -Recurse -Force -ErrorAction SilentlyContinue
}

if ((Test-Path $venvPython) -and (Test-PythonModule -Python $venvPython -Module "PyInstaller")) {
    Invoke-Checked -Command @($venvPython, "-m", "PyInstaller", "--noconfirm", "AIEQ.spec")
} else {
    Invoke-Checked -Command @("uv", "run", "--extra", "build", "pyinstaller", "--noconfirm", "AIEQ.spec")
}

Write-Host "Built: dist\AIEQ\AIEQ.exe"
