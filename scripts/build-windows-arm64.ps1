#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$All,
    [switch]$Mingw,
    [switch]$Msvc,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$DistDir   = Join-Path $RepoRoot 'dist'
$BuildDir  = Join-Path $RepoRoot 'build\windows'
$BuildDate = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Artifacts = New-Object System.Collections.Generic.List[string]

function Show-Usage {
@'
Usage: scripts\build-windows-arm64.ps1 [-All] [-Mingw] [-Msvc]

Build Windows ARM64 (aarch64) release archives:
  -All      build every supported ARM64 target available on this host (default)
  -Mingw    build a MinGW-w64 aarch64 static library package (cross compile)
  -Msvc     build an MSVC ARM64 static library package from a Visual Studio
            ARM64 developer shell

The MinGW target requires the aarch64-w64-mingw32-gcc cross compiler. It
produces a NEON-enabled libfunnelcake.a (the GCC __aarch64__ macro selects
the NEON kernel sources).

The MSVC target requires cl.exe and lib.exe configured for ARM64, usually by
running from "ARM64 Native Tools Command Prompt for VS" or an equivalent
Developer PowerShell with VSCMD_ARG_TGT_ARCH=arm64. The MSVC build includes
the NEON kernels (gated on _M_ARM64 in addition to __aarch64__), matching
the MinGW build's instruction-set coverage.
'@
}

if ($Help) { Show-Usage; return }

# default: build everything when no flag is given
if (-not $Mingw -and -not $Msvc -and -not $All) { $Mingw = $true; $Msvc = $true }
if ($All) { $Mingw = $true; $Msvc = $true }

function Test-Tool {
    param([string]$Name)
    [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Resolve-MakeTool {
    if (Test-Tool 'make')          { return 'make' }
    if (Test-Tool 'mingw32-make')  { return 'mingw32-make' }
    return $null
}

function Write-AsciiFile {
    param([string]$Path, [string]$Content)
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $enc)
}

function Get-Sha256Line {
    param([string]$Path)
    $hash = (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLower()
    $name = Split-Path -Leaf $Path
    return "$hash  $name`n"
}

function New-PackageDir {
    param(
        [string]$PackageDir,
        [string]$TargetArch,
        [string]$Compiler,
        [string]$LibraryPath,
        [string]$LibraryName
    )
    $dest = Join-Path $DistDir $PackageDir
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Path (Join-Path $dest 'include') -Force | Out-Null
    Copy-Item $LibraryPath (Join-Path $dest $LibraryName)
    Copy-Item (Join-Path $RepoRoot 'include\funnelcake.h') (Join-Path $dest 'include')
    Copy-Item (Join-Path $RepoRoot 'include\funnelcake_helpers.h') (Join-Path $dest 'include')
    Copy-Item (Join-Path $RepoRoot 'README.md')  $dest
    Copy-Item (Join-Path $RepoRoot 'INSTALL.md') $dest

    $info = @(
        'name=funnelcake'
        'target_os=windows'
        "base_distribution=$Compiler"
        "target_arch=$TargetArch"
        "compiler=$Compiler"
        'lto=0'
        "build_date=$BuildDate"
    ) -join "`n"
    Write-AsciiFile -Path (Join-Path $dest 'BUILD_INFO') -Content ($info + "`n")
}

function New-Zip {
    param([string]$PackageDir)
    $zip    = Join-Path $DistDir "$PackageDir.zip"
    $sha    = "$zip.sha256"
    $source = Join-Path $DistDir $PackageDir
    if (Test-Path $zip) { Remove-Item -Force $zip }
    if (Test-Path $sha) { Remove-Item -Force $sha }
    Compress-Archive -Path $source -DestinationPath $zip -Force
    Write-AsciiFile -Path $sha -Content (Get-Sha256Line $zip)
    $Artifacts.Add("$PackageDir.zip")        | Out-Null
    $Artifacts.Add("$PackageDir.zip.sha256") | Out-Null
}

function Invoke-MingwArm64Build {
    $arch     = 'aarch64'
    $cc       = 'aarch64-w64-mingw32-gcc'
    $arTool   = 'aarch64-w64-mingw32-ar'
    $package  = "funnelcake-windows-mingw-$arch"
    $makeTool = Resolve-MakeTool

    if (-not (Test-Tool $cc) -or -not (Test-Tool $arTool) -or -not $makeTool) {
        Write-Host "==> Skipping MinGW ${arch}: $cc, $arTool, and make are required"
        return
    }

    Write-Host "==> Building Windows MinGW $arch artifact"
    Push-Location $RepoRoot
    try {
        & $makeTool clean
        if ($LASTEXITCODE -ne 0) { throw "$makeTool clean failed" }
        & $makeTool lib "CC=$cc" "AR=$arTool" 'LTO=0' 'UNAME_S=Windows_NT' "UNAME_M=$arch"
        if ($LASTEXITCODE -ne 0) { throw "$makeTool lib failed" }
    }
    finally { Pop-Location }

    New-PackageDir -PackageDir $package -TargetArch $arch -Compiler 'mingw-w64' `
        -LibraryPath (Join-Path $RepoRoot 'libfunnelcake.a') -LibraryName 'libfunnelcake.a'
    New-Zip -PackageDir $package
}

function Test-MsvcArm64Target {
    $val = $env:VSCMD_ARG_TGT_ARCH
    if (-not $val) { $val = $env:PROCESSOR_ARCHITECTURE }
    return ($val -match '^(arm64|ARM64)$')
}

function Invoke-MsvcArm64Build {
    $commonSources = @(
        'src\funnelcake.c',
        'src\funnelcake_hdr.c',
        'src\log.c',
        'src\detect.c',
        'src\kernels_scalar.c',
        'src\kernels_hdr_scalar.c',
        'src\tonemap.c',
        'src\kernels_upscale_scalar.c',
        'src\bindings_support.c',
        'src\kernels_neon.c',
        'src\kernels_hdr_neon.c',
        'src\kernels_upscale_neon.c',
        'src\tonemap_neon.c'
    )

    if (-not (Test-Tool 'cl') -or -not (Test-Tool 'lib')) {
        Write-Host '==> Skipping MSVC: cl.exe and lib.exe are required'
        return
    }

    if (-not (Test-MsvcArm64Target)) {
        Write-Host '==> Skipping MSVC: this shell does not target ARM64 (set VSCMD_ARG_TGT_ARCH=arm64 or use the ARM64 Native Tools prompt)'
        return
    }

    $arch       = 'arm64'
    $package    = "funnelcake-windows-msvc-$arch"
    $objDir     = Join-Path $BuildDir "msvc-$arch"
    $libPath    = Join-Path $objDir 'funnelcake.lib'
    $includeDir = Join-Path $RepoRoot 'include'
    $srcDir     = Join-Path $RepoRoot 'src'

    Write-Host "==> Building Windows MSVC $arch artifact"
    if (Test-Path $objDir) { Remove-Item -Recurse -Force $objDir }
    New-Item -ItemType Directory -Path $objDir -Force | Out-Null

    $objs = @()
    foreach ($rel in $commonSources) {
        $srcPath = Join-Path $RepoRoot $rel
        $objPath = Join-Path $objDir ([IO.Path]::GetFileNameWithoutExtension($rel) + '.obj')
        & cl /nologo /std:c11 /O2 /W4 /WX `
            /D_CRT_SECURE_NO_WARNINGS /D_POSIX_C_SOURCE=200112L `
            "/I$includeDir" "/I$srcDir" `
            "/Fo$objPath" /c $srcPath
        if ($LASTEXITCODE -ne 0) { throw "cl failed for $rel" }
        $objs += $objPath
    }

    & lib /nologo "/OUT:$libPath" $objs
    if ($LASTEXITCODE -ne 0) { throw 'lib failed' }

    New-PackageDir -PackageDir $package -TargetArch $arch -Compiler 'msvc' `
        -LibraryPath $libPath -LibraryName 'funnelcake.lib'
    New-Zip -PackageDir $package
}

New-Item -ItemType Directory -Path $DistDir  -Force | Out-Null
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null

if ($Mingw) { Invoke-MingwArm64Build }
if ($Msvc)  { Invoke-MsvcArm64Build }

$finalMake = Resolve-MakeTool
if ($finalMake) {
    Push-Location $RepoRoot
    try { & $finalMake clean | Out-Null } finally { Pop-Location }
}

if ($Artifacts.Count -eq 0) {
    Write-Host ''
    Write-Host 'No Windows ARM64 artifacts were built.'
    Write-Host 'Install aarch64-w64-mingw32-gcc, or run -Msvc from an ARM64 Visual Studio developer shell.'
    exit 1
}

Write-Host ''
Write-Host "Artifacts written to ${DistDir}:"
foreach ($a in $Artifacts) { Write-Host "  $a" }
