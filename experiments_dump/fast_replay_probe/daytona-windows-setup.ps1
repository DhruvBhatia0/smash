$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$root = "C:\slippi-probe"
$zip = Join-Path $root "slippi.zip"
$dolphin = Join-Path $root "dolphin"
New-Item -ItemType Directory -Force -Path $root | Out-Null

if (-not (Test-Path (Join-Path $dolphin "Slippi Dolphin.exe"))) {
    Invoke-WebRequest `
        -Uri "https://github.com/project-slippi/Ishiiruka/releases/download/v3.6.4/FM-Slippi-3.6.4-Win.zip" `
        -OutFile $zip
    if (Test-Path $dolphin) { Remove-Item -Recurse -Force $dolphin }
    Expand-Archive -Path $zip -DestinationPath $dolphin
}

if (-not (Test-Path "C:\Windows\System32\vcruntime140.dll")) {
    $redist = Join-Path $root "vc_redist.x64.exe"
    Invoke-WebRequest -Uri "https://aka.ms/vs/16/release/vc_redist.x64.exe" -OutFile $redist
    $installer = Start-Process -FilePath $redist `
        -ArgumentList "/install", "/quiet", "/norestart" -Wait -PassThru
    if ($installer.ExitCode -notin 0, 1638, 3010) {
        throw "Visual C++ runtime installer failed with exit code $($installer.ExitCode)"
    }
}

Get-ChildItem -Path $dolphin -Recurse -Filter "*.exe" |
    Select-Object FullName, Length |
    Format-Table -AutoSize

Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors |
    Format-List
Get-CimInstance Win32_VideoController |
    Select-Object Name, DriverVersion, AdapterRAM |
    Format-List
