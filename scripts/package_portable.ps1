param(
    [string]$Version = "dev",
    [string[]]$Model = @("Qwen3-4B-Q4_K_M.gguf"),
    [switch]$IncludeAllModels,
    [switch]$NoModels,
    [switch]$NoRuntime,
    [switch]$NoCurves,
    [string]$VbCableInstallerPath = "",
    [switch]$SkipBuild,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

function Copy-DirectoryFresh {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (-not (Test-Path $Source)) {
        return
    }
    Remove-Item -LiteralPath $Destination -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

function Copy-FileIfExists {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (Test-Path $Source) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Write-PortableReadme {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Version
    )

    @"
AIEQ portable package ($Version)

1. Run AIEQ.exe.
2. If audio routing does not work yet, install VB-Cable manually from installers\ if this package contains it.
3. Keep these folders next to AIEQ.exe:
   - models\        GGUF model files
   - runtime\       llama.cpp runtime with llama-server.exe and CUDA DLLs
   - curves\        device and target curves
4. The default .env expects:
   models\Qwen3-4B-Q4_K_M.gguf
   runtime\llama.cpp\llama-server.exe
5. For NVIDIA GPU acceleration, install a recent NVIDIA driver and make sure llama-server sees CUDA0:
   runtime\llama.cpp\llama-server.exe --list-devices

VB-Cable is a third-party audio driver and usually requires administrator rights and a reboot.
"@ | Set-Content -LiteralPath $Path -Encoding UTF8
}

function New-ZipArchive {
    param(
        [Parameter(Mandatory = $true)][string]$SourceFolder,
        [Parameter(Mandatory = $true)][string]$ArchivePath
    )

    $sevenZip = Get-Command "7z.exe" -ErrorAction SilentlyContinue
    if ($sevenZip) {
        & $sevenZip.Source a -tzip -mx=1 $ArchivePath $SourceFolder
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip failed with exit code $LASTEXITCODE"
        }
        return
    }

    $largeFiles = Get-ChildItem -LiteralPath $SourceFolder -Recurse -File | Where-Object { $_.Length -gt 1900MB }
    if ($largeFiles) {
        Write-Warning "Package contains files larger than 1.9GB. Install 7-Zip and rerun with -Zip, or upload the folder contents manually."
        return
    }

    Compress-Archive -LiteralPath $SourceFolder -DestinationPath $ArchivePath -Force
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not $SkipBuild) {
    & "$PSScriptRoot\build_exe.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "build_exe.ps1 failed with exit code $LASTEXITCODE"
    }
}

$builtApp = Join-Path $projectRoot "dist\AIEQ"
if (-not (Test-Path (Join-Path $builtApp "AIEQ.exe"))) {
    throw "Built app was not found: dist\AIEQ\AIEQ.exe"
}

$releaseRoot = Join-Path $projectRoot "release"
$safeVersion = ($Version -replace '[^\w\.-]+', '-').Trim("-")
if (-not $safeVersion) {
    $safeVersion = "dev"
}
$packageName = "AIEQ-$safeVersion-portable"
$packageRoot = Join-Path $releaseRoot $packageName

Remove-Item -LiteralPath $packageRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null
Copy-Item -Path (Join-Path $builtApp "*") -Destination $packageRoot -Recurse -Force

Copy-FileIfExists -Source ".env.example" -Destination (Join-Path $packageRoot ".env")
Copy-FileIfExists -Source ".env.example" -Destination (Join-Path $packageRoot ".env.example")

if (-not $NoCurves) {
    Copy-DirectoryFresh -Source "curves" -Destination (Join-Path $packageRoot "curves")
}

if (-not $NoRuntime) {
    Copy-DirectoryFresh -Source "runtime" -Destination (Join-Path $packageRoot "runtime")
}

if (-not $NoModels) {
    $modelsDestination = Join-Path $packageRoot "models"
    New-Item -ItemType Directory -Force -Path $modelsDestination | Out-Null
    if ($IncludeAllModels) {
        Get-ChildItem -LiteralPath "models" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension.ToLowerInvariant() -in @(".gguf", ".bin", ".safetensors") } |
            Copy-Item -Destination $modelsDestination -Force
    } else {
        foreach ($modelName in $Model) {
            Copy-FileIfExists -Source (Join-Path "models" $modelName) -Destination (Join-Path $modelsDestination $modelName)
        }
    }
}

if ($VbCableInstallerPath) {
    if (-not (Test-Path $VbCableInstallerPath)) {
        throw "VB-Cable installer was not found: $VbCableInstallerPath"
    }
    $installerDestination = Join-Path $packageRoot ("installers\" + (Split-Path -Leaf $VbCableInstallerPath))
    Copy-FileIfExists -Source $VbCableInstallerPath -Destination $installerDestination
}

Write-PortableReadme -Path (Join-Path $packageRoot "PORTABLE_README.txt") -Version $safeVersion

if ($Zip) {
    $archivePath = Join-Path $releaseRoot "$packageName.zip"
    Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
    New-ZipArchive -SourceFolder $packageRoot -ArchivePath $archivePath
}

Write-Host "Portable package: $packageRoot"
if ($Zip -and (Test-Path (Join-Path $releaseRoot "$packageName.zip"))) {
    Write-Host "Archive: $(Join-Path $releaseRoot "$packageName.zip")"
}
