param(
    [ValidateSet("D3D", "OGL")]
    [string]$Backend = "D3D",
    [switch]$DumpFrames
)

$ErrorActionPreference = "Stop"
$root = "C:\slippi-probe"
$dolphin = Join-Path $root "dolphin\Slippi Dolphin.exe"
$user = Join-Path $root ("user-" + $Backend.ToLowerInvariant())
$config = Join-Path $user "Config"
$dump = Join-Path $user "Dump\Frames"
$log = Join-Path $root ("windows-" + $Backend.ToLowerInvariant() + ".log")
$errorLog = Join-Path $root ("windows-" + $Backend.ToLowerInvariant() + ".error.log")
$result = Join-Path $root ("windows-" + $Backend.ToLowerInvariant() + ".json")
$iso = Join-Path $root "melee.iso"
$slp = Join-Path $root "realtimeTest.slp"
$playback = Join-Path $root "playback.json"

$uploadedIso = Get-ChildItem -Path "C:\Users" -Filter "*.iso" -File -Recurse | Select-Object -First 1
$uploadedSlp = Get-ChildItem -Path "C:\Users" -Filter "*.slp" -File -Recurse | Select-Object -First 1
if (-not (Test-Path $iso)) {
    if ($null -eq $uploadedIso) { throw "Uploaded ISO was not found under C:\Users" }
    Copy-Item -Path $uploadedIso.FullName -Destination $iso
}
if (-not (Test-Path $slp)) {
    if ($null -eq $uploadedSlp) { throw "Uploaded SLP was not found under C:\Users" }
    Copy-Item -Path $uploadedSlp.FullName -Destination $slp
}

New-Item -ItemType Directory -Force -Path $config, $dump | Out-Null
Remove-Item -Force -ErrorAction SilentlyContinue $log, $errorLog, $result
Get-ChildItem -Path $dump -File -ErrorAction SilentlyContinue | Remove-Item -Force

@{
    replay = $slp
    startFrame = -123
    endFrame = 2182
    commandId = "daytona-windows-cpu-unbounded"
    shouldResync = $false
    rollbackDisplayMethod = "off"
} | ConvertTo-Json -Compress | Set-Content -Encoding ASCII $playback

$dumpValue = if ($DumpFrames) { "True" } else { "False" }
@"
[Core]
DefaultISO = $iso
EmulationSpeed = 0.00000000
CPUCore = 1
CPUThread = True
Fastmem = True
GFXBackend = $Backend

[Display]
Fullscreen = False
RenderToMain = True
RenderWindowWidth = 960
RenderWindowHeight = 720

[Movie]
DumpFrames = $dumpValue
DumpFramesSilent = True

[DSP]
Backend = Null
DumpAudio = False
DumpAudioSilent = True

[Interface]
ShowLogWindow = False
ShowLogConfigWindow = False
"@ | Set-Content -Encoding ASCII (Join-Path $config "Dolphin.ini")

@"
[Settings]
ShowFPS = False
DumpFramesAsImages = False
InternalResolutionFrameDumps = False
EFBScale = 2
UseFFV1 = False
DumpCodec = rawvideo
DumpFormat = avi
BitrateKbps = 2500
"@ | Set-Content -Encoding ASCII (Join-Path $config "GFX.ini")

@"
[Options]
Verbosity = 1
WriteToConsole = True
WriteToFile = True
WriteToWindow = False

[Logs]
MASTER = True
VIDEO = True
BOOT = True
OSREPORT = True
"@ | Set-Content -Encoding ASCII (Join-Path $config "Logger.ini")

$arguments = @(
    "-u", ('"' + $user + '"'),
    "-i", ('"' + $playback + '"'),
    "-e", ('"' + $iso + '"'),
    "--hide-seekbar", "--cout", "--batch", "-v", $Backend
)
$clock = [Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $dolphin -ArgumentList $arguments -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $errorLog

$targetReached = $false
while (-not $process.HasExited -and $clock.Elapsed.TotalSeconds -lt 180) {
    Start-Sleep -Milliseconds 250
    if (Test-Path $log) {
        $targetReached = [bool](Select-String -Path $log -Pattern "\[CURRENT_FRAME\] 218[2-9]" -Quiet)
        if ($targetReached) { break }
    }
    $process.Refresh()
}

if (-not $process.HasExited) {
    $null = $process.CloseMainWindow()
    if (-not $process.WaitForExit(5000)) { Stop-Process -Id $process.Id -Force }
}
$process.WaitForExit()
$clock.Stop()

$frames = @()
if (Test-Path $log) {
    $frames = Select-String -Path $log -Pattern "\[CURRENT_FRAME\] (-?\d+)" | ForEach-Object {
        [int]$_.Matches[0].Groups[1].Value
    }
}
$videos = @(Get-ChildItem -Path $dump -File -ErrorAction SilentlyContinue | ForEach-Object {
    @{ path = $_.FullName; bytes = $_.Length }
})
$output = [ordered]@{
    backend = $Backend
    cpu = (Get-CimInstance Win32_Processor).Name.Trim()
    logicalProcessors = (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors
    videoController = (Get-CimInstance Win32_VideoController).Name
    dumpFrames = [bool]$DumpFrames
    targetReached = $targetReached
    wallSeconds = [Math]::Round($clock.Elapsed.TotalSeconds, 3)
    currentFrameLogEntries = $frames.Count
    firstCurrentFrame = if ($frames.Count) { $frames[0] } else { $null }
    lastCurrentFrame = if ($frames.Count) { $frames[-1] } else { $null }
    videos = $videos
}
$output | ConvertTo-Json -Depth 4 | Set-Content -Encoding ASCII $result
$output | ConvertTo-Json -Depth 4
