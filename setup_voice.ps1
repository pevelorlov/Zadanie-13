param(
    [switch]$SkipDownloads
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$voiceRoot = Join-Path $projectRoot "tools\whisper"
$sourceRoot = Join-Path $voiceRoot "source"
$buildRoot = Join-Path $voiceRoot "build-vulkan"
$binRoot = Join-Path $voiceRoot "bin"
$ffmpegRoot = Join-Path $projectRoot "tools\ffmpeg"
$modelRoot = Join-Path $projectRoot "models"
$modelPath = Join-Path $modelRoot "ggml-large-v3-turbo.bin"
$cacheRoot = Join-Path $projectRoot "tools\downloads"

New-Item -ItemType Directory -Force -Path $voiceRoot, $binRoot, $ffmpegRoot, $modelRoot, $cacheRoot | Out-Null

$sdkRoot = Get-ChildItem -LiteralPath "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $sdkRoot -or -not (Test-Path -LiteralPath (Join-Path $sdkRoot.FullName "Bin\glslc.exe"))) {
    throw "Vulkan SDK was not found. Install KhronosGroup.VulkanSDK and try again."
}
$env:VULKAN_SDK = $sdkRoot.FullName
$env:Path = (Join-Path $sdkRoot.FullName "Bin") + ";" + $env:Path

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Visual Studio Build Tools (vswhere.exe) was not found."
}
$visualStudio = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $visualStudio) {
    throw "The Visual Studio C++ build tools are not installed."
}
$developerCommand = Join-Path $visualStudio "Common7\Tools\VsDevCmd.bat"
$cmake = Join-Path $visualStudio "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = Join-Path $visualStudio "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path -LiteralPath $cmake) -or -not (Test-Path -LiteralPath $ninja)) {
    throw "CMake or Ninja is missing from Visual Studio Build Tools."
}

if (-not $SkipDownloads) {
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
        Write-Host "Downloading official whisper.cpp..."
        & git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git $sourceRoot
        if ($LASTEXITCODE -ne 0) { throw "Could not download whisper.cpp." }
    }

    if (-not (Test-Path -LiteralPath $modelPath)) {
        Write-Host "Downloading large-v3-turbo model (about 1.5 GB)..."
        $partialModel = "$modelPath.part"
        & curl.exe -L --fail --retry 3 --output $partialModel "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
        if ($LASTEXITCODE -ne 0) { throw "Could not download the Whisper model." }
        $modelHash = (Get-FileHash -LiteralPath $partialModel -Algorithm SHA1).Hash.ToLowerInvariant()
        if ($modelHash -ne "4af2b29d7ec73d781377bfd1758ca957a807e941") {
            throw "Model checksum mismatch. Actual value: $modelHash"
        }
        Move-Item -LiteralPath $partialModel -Destination $modelPath -Force
    }

    if (-not (Test-Path -LiteralPath (Join-Path $ffmpegRoot "bin\ffmpeg.exe"))) {
        Write-Host "Downloading portable FFmpeg..."
        $ffmpegZip = Join-Path $cacheRoot "ffmpeg-release-essentials.zip"
        $ffmpegExtract = Join-Path $cacheRoot "ffmpeg-release-essentials"
        & curl.exe -L --fail --retry 3 --output $ffmpegZip "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        if ($LASTEXITCODE -ne 0) { throw "Could not download FFmpeg." }
        Expand-Archive -LiteralPath $ffmpegZip -DestinationPath $ffmpegExtract -Force
        $ffmpegExe = Get-ChildItem -LiteralPath $ffmpegExtract -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
        if (-not $ffmpegExe) { throw "ffmpeg.exe was not found in the downloaded archive." }
        $distributionRoot = Split-Path -Parent (Split-Path -Parent $ffmpegExe.FullName)
        Copy-Item -Path (Join-Path $distributionRoot "*") -Destination $ffmpegRoot -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "CMakeLists.txt"))) {
    throw "whisper.cpp sources are missing. Run without -SkipDownloads."
}

Write-Host "Building whisper-server with Vulkan..."
$buildCommand = '"{0}" -arch=x64 && "{1}" -S "{2}" -B "{3}" -G Ninja -DGGML_VULKAN=ON -DWHISPER_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="{4}" && "{1}" --build "{3}" --target whisper-server --config Release' -f $developerCommand, $cmake, $sourceRoot, $buildRoot, $ninja
& cmd.exe /d /s /c $buildCommand
if ($LASTEXITCODE -ne 0) { throw "whisper-server build failed." }

$serverExe = Get-ChildItem -LiteralPath $buildRoot -Recurse -Filter "whisper-server.exe" | Select-Object -First 1
if (-not $serverExe) { throw "whisper-server.exe was not found after the build." }
$runtimeFolder = $serverExe.Directory.FullName
Copy-Item -LiteralPath $serverExe.FullName -Destination $binRoot -Force
Get-ChildItem -LiteralPath $runtimeFolder -Filter "*.dll" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $binRoot -Force
}

Write-Host ""
Write-Host "Done. Local voice input is installed." -ForegroundColor Green
Write-Host "Whisper server: $(Join-Path $binRoot 'whisper-server.exe')"
Write-Host "Model: $modelPath"
Write-Host "FFmpeg: $(Join-Path $ffmpegRoot 'bin\ffmpeg.exe')"
Write-Host "Restart the app using run.bat."
