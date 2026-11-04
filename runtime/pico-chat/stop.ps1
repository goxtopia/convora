param([switch]$KeepBoard,[string]$AdbPath='adb',[string]$Serial=$env:ANDROID_SERIAL)
$ErrorActionPreference='Stop'
$saved=Join-Path $PSScriptRoot 'runtime/server.json'
if (Test-Path -LiteralPath $saved) {
    $state=Get-Content -Raw -LiteralPath $saved | ConvertFrom-Json
    $process=Get-CimInstance Win32_Process -Filter ('ProcessId='+[int]$state.pid) -ErrorAction SilentlyContinue
    $script=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'server.py'))
    if ($process -and $process.CommandLine -and $process.CommandLine.IndexOf($script,[StringComparison]::OrdinalIgnoreCase) -ge 0) { Stop-Process -Id $state.pid }
    Remove-Item -LiteralPath $saved
}
if (-not $KeepBoard) {
    if (-not $Serial) { throw 'Pass -Serial or set ANDROID_SERIAL to the intended board.' }
    & $AdbPath -s $Serial shell '/userdata/pico-chat/S98picochat stop'
}
Write-Output 'Pico Chat 已停止。'
