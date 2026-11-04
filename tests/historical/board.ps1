param([string[]]$Variants=@('trace'),[string]$Tag='trace',[switch]$ServiceAlreadyStopped)
$ErrorActionPreference='Stop'
$adb='D:\platform-tools\adb.exe';$serial='b9deb5cb2fb52c4f'
$project=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$base='/mnt/sdcard/pico-models/qat-toolchain-match-20261029'
function A([string[]]$ArgList){$r=& $adb -s $serial @ArgList;if($LASTEXITCODE){throw ('ADB failed '+($ArgList -join ' '))};return $r}
function Health{return ((A @('shell','http_proxy= https_proxy= all_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= wget -qO- http://127.0.0.1:8781/health')|Out-String).Trim()|ConvertFrom-Json)}
$before=$null
if(-not $ServiceAlreadyStopped){
 $before=Health
 if(-not $before.ready -or $before.checkpoint_sha256 -ne '9db7a2fa06b2687bfaa8091b9814bbf6ce52c7aba64d45bd66af933a902adef5'){throw 'Unexpected live model'}
}
$dest=Join-Path $PSScriptRoot ('board-'+$Tag)
if(Test-Path $dest){throw 'Evidence directory exists'}
[IO.Directory]::CreateDirectory($dest)|Out-Null
$before|ConvertTo-Json -Depth 7|Set-Content (Join-Path $dest 'before.json')
foreach($v in $Variants){
 if($v -notmatch '^[a-z0-9-]+$'){throw 'Bad variant'}
 $board=$base+'/'+$Tag+'-'+$v
 $r=A @('shell',('test ! -e '+$board+' && mkdir -p '+$board+' && echo CREATED'))|Out-String
 if(-not $r.Contains('CREATED')){throw 'Board directory exists'}
 foreach($name in @('model.rknn','inputs.f32','shapes.txt')){
  $local=Join-Path $PSScriptRoot ($v+'/'+$name);A @('push',$local,($board+'/'+$name))|Out-Null
  $h=(A @('shell',('sha256sum '+$board+'/'+$name))|Out-String).Trim().Split(' ')[0]
  if($h -ne (Get-FileHash $local -Algorithm SHA256).Hash.ToLowerInvariant()){throw 'Payload SHA mismatch'}
 }
 A @('push',(Join-Path $PSScriptRoot 'runner'),($board+'/runner'))|Out-Null
 A @('shell',('chmod 755 '+$board+'/runner'))|Out-Null
}
$web=$null
try{$web=Invoke-RestMethod http://127.0.0.1:9110/api/health -NoProxy -TimeoutSec 3}catch{}
if($web -and ($web.queue.busy -or $web.queue.waiting)){throw 'Chat is busy; staged diagnostics only'}
$restore=$false
try{
 if(-not $ServiceAlreadyStopped){
  $now=Health;if($now.pid -ne $before.pid){throw 'Service changed'}
  $restore=$true
  if($web){& (Join-Path $project 'pico-chat/stop.ps1')}else{A @('shell','/userdata/pico-chat/S98picochat stop')|Out-Null}
 }
 $dma=A @('shell','cat /proc/rk_dma_heap/alloc_bitmap')|Out-String
 if($dma -notmatch 'Used:\s+12 KiB'){throw 'DMA is not idle'}
 foreach($v in $Variants){
  $board=$base+'/'+$Tag+'-'+$v;$local=Join-Path $dest $v;[IO.Directory]::CreateDirectory($local)|Out-Null
  A @('shell',('cd '+$board+' && { ./runner > run.log 2>&1; echo $? > exit-code.txt; }'))|Out-Null
  foreach($file in @('run.log','exit-code.txt','actual.f32')){
   $exists=(A @('shell',('test -f '+$board+'/'+$file+' && echo EXISTS'))|Out-String).Trim()
   if($exists -eq 'EXISTS'){A @('pull',($board+'/'+$file),(Join-Path $local $file))|Out-Null}
  }
  if((Get-Content (Join-Path $local 'exit-code.txt') -Raw).Trim() -ne '0'){throw ('Probe failed '+$v)}
 }
}finally{
 if($restore){
  if($web){& (Join-Path $project 'pico-chat/start.ps1')}else{A @('shell','/userdata/pico-chat/S98picochat start')|Out-Null}
  $after=Health;$after|ConvertTo-Json -Depth 7|Set-Content (Join-Path $dest 'after.json')
  if(-not $after.ready -or $after.checkpoint_sha256 -ne $before.checkpoint_sha256 -or $after.context_tokens -ne $before.context_tokens){throw 'Live model restore failed'}
  A @('shell','cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /proc/rk_dma_heap/alloc_bitmap')|Set-Content (Join-Path $dest 'restored.txt')
 }
}
