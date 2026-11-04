param([int]$Port=9110,[int]$BoardPort=18765,[string]$ListenHost='0.0.0.0',[string]$AdbPath='adb',[string]$Serial=$env:ANDROID_SERIAL)
$ErrorActionPreference='Stop'
$adb=$AdbPath
$serial=$Serial
if (-not $serial) { throw 'Pass -Serial or set ANDROID_SERIAL to the intended board.' }
$python=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$runtime=Join-Path $PSScriptRoot 'runtime'
[IO.Directory]::CreateDirectory($runtime) | Out-Null
& $adb -s $serial get-state | Out-Null
if ($LASTEXITCODE) { throw '请连接 Luckfox Pico 的 USB，并检查 adb devices。' }
& $adb -s $serial shell '/userdata/pico-chat/S98picochat start'
$forwardList=& $adb forward --list
$expectedForward='{0} tcp:{1} tcp:8781' -f $serial,$BoardPort
$knownForwards=@($forwardList | ForEach-Object { $_.Trim() -replace '\s+',' ' })
if ($knownForwards -notcontains $expectedForward) {
    & $adb -s $serial forward --no-rebind ('tcp:'+$BoardPort) tcp:8781
    if ($LASTEXITCODE) { throw '板端转发端口已被其他服务使用。' }
}
$url='http://127.0.0.1:'+ $Port
$lanInterfaces=@(Get-NetRoute -AddressFamily IPv4 -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty InterfaceIndex)
$physicalInterfaces=@(Get-NetAdapter -Physical -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ifIndex)
$lanUrls=@(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $lanInterfaces -contains $_.InterfaceIndex -and $physicalInterfaces -contains $_.InterfaceIndex -and $_.IPAddress -match '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)' } | ForEach-Object { 'http://'+$_.IPAddress+':'+$Port } | Select-Object -Unique)
$server=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'server.py'))
try {
    $existing=Invoke-RestMethod ($url+'/api/health') -TimeoutSec 2 -NoProxy
    $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+[int]$existing.web_pid)
    if ($owner.CommandLine -and $owner.CommandLine.IndexOf($server,[StringComparison]::OrdinalIgnoreCase) -ge 0) {
        @{pid=$existing.web_pid;port=$Port;board_port=$BoardPort;url=$url} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runtime 'server.json')
        Write-Output ('Pico Chat: '+$url);$lanUrls | ForEach-Object { Write-Output ('LAN: '+$_) };return
    }
} catch {}
$probe=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Parse($ListenHost),$Port)
$probe.Server.ExclusiveAddressUse=$true
try { $probe.Start() } catch { throw ('网页端口 '+$Port+' 已被其他服务使用。') } finally { $probe.Stop() }
$process=Start-Process -FilePath $python -ArgumentList @('-u',('"'+$server+'"'),'--host',$ListenHost,'--port',[string]$Port,'--board-port',[string]$BoardPort) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtime 'web.log') -RedirectStandardError (Join-Path $runtime 'web-error.log')
@{pid=$process.Id;port=$Port;board_port=$BoardPort;url=$url} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runtime 'server.json')
for ($attempt=0;$attempt -lt 30;$attempt++) {
    Start-Sleep -Milliseconds 200
    try {
        $health=Invoke-RestMethod ($url+'/api/health') -TimeoutSec 2 -NoProxy
        $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+[int]$health.web_pid)
        if ($owner.CommandLine -and $owner.CommandLine.IndexOf($server,[StringComparison]::OrdinalIgnoreCase) -ge 0 -and $health.online) {
            @{pid=$health.web_pid;launcher_pid=$process.Id;port=$Port;board_port=$BoardPort;url=$url;listen_host=$ListenHost;lan_urls=$lanUrls} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runtime 'server.json')
            Write-Output ('Pico Chat: '+$url);$lanUrls | ForEach-Object { Write-Output ('LAN: '+$_) };return
        }
    } catch {}
    if ($process.HasExited) { throw ('网页服务启动失败，请查看 '+(Join-Path $runtime 'web-error.log')) }
}
throw ('服务未就绪，请检查USB及 '+(Join-Path $runtime 'web-error.log'))
