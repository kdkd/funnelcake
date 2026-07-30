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
Usage: scripts\build-windows.ps1 [-All] [-Mingw] [-Msvc]

Build Windows release archives:
  -All      build every supported Windows target available on this host (default)
  -Mingw    build MinGW-w64 static library packages
  -Msvc     build an MSVC static library package from a Visual Studio shell

MinGW targets require cross compilers such as x86_64-w64-mingw32-gcc.
The MSVC target requires cl.exe and lib.exe in PATH, usually by running from
"x64 Native Tools Command Prompt for VS" or an equivalent Developer PowerShell.
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
    $Artifacts.Add("$PackageDir.zip")      | Out-Null
    $Artifacts.Add("$PackageDir.zip.sha256") | Out-Null
}

function Invoke-MingwBuild {
    param([string]$Arch, [string]$CC, [string]$ARTool)
    $package  = "funnelcake-windows-mingw-$Arch"
    $makeTool = Resolve-MakeTool

    if (-not (Test-Tool $CC) -or -not (Test-Tool $ARTool) -or -not $makeTool) {
        Write-Host "==> Skipping MinGW ${Arch}: $CC, $ARTool, and make are required"
        return
    }

    Write-Host "==> Building Windows MinGW $Arch artifact"
    Push-Location $RepoRoot
    try {
        & $makeTool clean
        if ($LASTEXITCODE -ne 0) { throw "$makeTool clean failed" }
        & $makeTool lib "CC=$CC" "AR=$ARTool" 'LTO=0' 'UNAME_S=Windows_NT' "UNAME_M=$Arch"
        if ($LASTEXITCODE -ne 0) { throw "$makeTool lib failed" }
    }
    finally { Pop-Location }

    New-PackageDir -PackageDir $package -TargetArch $Arch -Compiler 'mingw-w64' `
        -LibraryPath (Join-Path $RepoRoot 'libfunnelcake.a') -LibraryName 'libfunnelcake.a'
    New-Zip -PackageDir $package
}

function Get-MsvcArchName {
    $val = $env:VSCMD_ARG_TGT_ARCH
    if (-not $val) { $val = $env:PROCESSOR_ARCHITECTURE }
    switch -Regex ($val) {
        '^(x64|AMD64|amd64)$' { return 'x64' }
        '^(arm64|ARM64)$'     { return 'arm64' }
        default               { return 'unknown' }
    }
}

function Invoke-MsvcBuild {
    $commonSources = @(
        'src\funnelcake.c',
        'src\funnelcake_hdr.c',
        'src\log.c',
        'src\detect.c',
        'src\kernels_scalar.c',
        'src\kernels_hdr_scalar.c',
        'src\tonemap.c',
        'src\kernels_upscale_scalar.c',
        'src\bindings_support.c'
    )

    if (-not (Test-Tool 'cl') -or -not (Test-Tool 'lib')) {
        Write-Host '==> Skipping MSVC: cl.exe and lib.exe are required'
        return
    }

    $arch = Get-MsvcArchName
    if ($arch -eq 'unknown') {
        Write-Host '==> Skipping MSVC: unable to determine Visual Studio target architecture'
        return
    }

    if ($arch -eq 'arm64') {
        $commonSources += @(
            'src\kernels_neon.c',
            'src\kernels_hdr_neon.c',
            'src\kernels_upscale_neon.c',
            'src\tonemap_neon.c'
        )
    }

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

if ($Mingw) {
    Invoke-MingwBuild -Arch 'x86_64'  -CC 'x86_64-w64-mingw32-gcc'  -ARTool 'x86_64-w64-mingw32-ar'
    Invoke-MingwBuild -Arch 'aarch64' -CC 'aarch64-w64-mingw32-gcc' -ARTool 'aarch64-w64-mingw32-ar'
}

if ($Msvc) {
    Invoke-MsvcBuild
}

$finalMake = Resolve-MakeTool
if ($finalMake) {
    Push-Location $RepoRoot
    try { & $finalMake clean | Out-Null } finally { Pop-Location }
}

if ($Artifacts.Count -eq 0) {
    Write-Host ''
    Write-Host 'No Windows artifacts were built.'
    Write-Host 'Install MinGW-w64, or run -Msvc from a Visual Studio developer shell.'
    exit 1
}

Write-Host ''
Write-Host "Artifacts written to ${DistDir}:"
foreach ($a in $Artifacts) { Write-Host "  $a" }
