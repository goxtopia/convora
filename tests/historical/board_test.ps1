param([string[]]$Variants=@('fp-ptq','qat-weights-ptq','qat-qdq'),[int]$Steps=2048)
$ErrorActionPreference='Stop'
$adb='D:\platform-tools\adb.exe';$serial='b9deb5cb2fb52c4f'
$project=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$base='/mnt/sdcard/pico-models/delta-sft-qat-20261029'
$healthCommand='http_proxy= https_proxy= all_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= wget -qO- http://127.0.0.1:8781/health'
function Adb([string[]]$Command) {
    $answer=& $adb -s $serial @Command
    if ($LASTEXITCODE -ne 0) { throw ('ADB failed: '+($Command -join ' ')) }
    return $answer
}
function Health { return ((Adb @('shell',$healthCommand)|Out-String).Trim()|ConvertFrom-Json) }
$before=Health
if (-not $before.ready) { throw 'Original board daemon is not ready' }
$exe=(Adb @('shell',('readlink /proc/'+[int]$before.pid+'/exe'))|Out-String).Trim()
if ($exe -ne '/userdata/pico-chat/pico-chatd') { throw 'Unexpected board service process' }
$webBefore=$null
try {
    $candidate=Invoke-RestMethod http://127.0.0.1:9110/api/health -TimeoutSec 2 -NoProxy
    $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+[int]$candidate.web_pid)
    if ($owner.CommandLine -and $owner.CommandLine.Contains((Join-Path $project 'pico-chat/server.py'))) { $webBefore=$candidate }
} catch {}
$results=Join-Path $PSScriptRoot 'board-results'
[IO.Directory]::CreateDirectory($results)|Out-Null
@{board=$before;web_was_running=($null -ne $webBefore);web=$webBefore}|ConvertTo-Json -Depth 8|Set-Content -Encoding utf8NoBOM (Join-Path $results 'original-service.json')
# Stage and verify every payload before interrupting the original daemon.
foreach ($variant in $Variants) {
    if ($variant -notmatch '^[a-z0-9-]+$') { throw 'Invalid variant' }
    $local=Join-Path $PSScriptRoot $variant;$board=$base+'/'+$variant
    foreach ($name in @('data/embedding.f32','data/memory.i8','data/memory_scale.f32','data/sequence.i32','data/reference.f32','data/state-reference.f32','models/tcn_engram.rknn','delta-bench')) {
        if (-not (Test-Path -LiteralPath (Join-Path $local $name))) { throw ('Missing '+$variant+'/'+$name) }
    }
    $created=Adb @('shell',('test ! -e '+$board+' && mkdir -p '+$board+'/data '+$board+'/models '+$board+'/results && echo DIRECTORY_READY'))|Out-String
    if (-not $created.Contains('DIRECTORY_READY')) { throw ('Refusing to reuse existing board directory '+$board) }
    if ($variant -eq 'qat-qdq') {
        foreach ($name in @('delta-dump','data/diag-sequence.i32','data/diag-flags.i32')) {
            $file=Join-Path $local $name
            Adb @('push',$file,($board+'/'+$name))|Out-Null
            $actual=(Adb @('shell',('sha256sum '+$board+'/'+$name))|Out-String).Trim().Split(' ')[0]
            if ($actual -ne (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()) { throw ('Board SHA mismatch '+$name) }
        }
        Adb @('shell',('chmod 755 '+$board+'/delta-dump'))|Out-Null
    }
    foreach ($name in @('data/embedding.f32','data/memory.i8','data/memory_scale.f32','data/sequence.i32','data/reference.f32','data/state-reference.f32','models/tcn_engram.rknn','delta-bench')) {
        $file=Join-Path $local $name;Adb @('push',$file,($board+'/'+$name))|Out-Null
        $actual=(Adb @('shell',('sha256sum '+$board+'/'+$name))|Out-String).Trim().Split(' ')[0]
        if ($actual -ne (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()) { throw ('Board SHA mismatch '+$name) }
    }
    Adb @('push',(Join-Path $project 'research/deltanet-npu-20261028/run_board.sh'),($board+'/run_board.sh'))|Out-Null
    $ready=Adb @('shell',('chmod 755 '+$board+'/delta-bench && test -x '+$board+'/delta-bench && echo BINARY_READY'))|Out-String
    if (-not $ready.Contains('BINARY_READY')) { throw 'Board binary is not executable' }
}
$restore=$false
try {
    # Recheck identity immediately before stopping only this known service.
    $current=Health
    if ($current.checkpoint_sha256 -ne $before.checkpoint_sha256 -or $current.pid -ne $before.pid) { throw 'Original daemon changed during staging' }
    $restore=$true
    if ($null -ne $webBefore) { & (Join-Path $project 'pico-chat/stop.ps1') }
    else { Adb @('shell','/userdata/pico-chat/S98picochat stop')|Out-Null }
    $idle=''
    for ($attempt=0;$attempt -lt 20;$attempt++) {
        $idle=Adb @('shell','cat /proc/rk_dma_heap/alloc_bitmap')|Out-String
        if ($idle -match 'Used:\s+12 KiB') { break }
        Start-Sleep -Milliseconds 200
    }
    $idle|Set-Content (Join-Path $results 'idle-dma.txt')
    if ($idle -notmatch 'Used:\s+12 KiB') { throw 'Original service stopped but NPU DMA is not idle; other allocations left untouched' }
    foreach ($variant in $Variants) {
        $board=$base+'/'+$variant
        foreach ($count in @(32,$Steps)) {
            $dest=Join-Path $results ($variant+'/steps-'+$count);[IO.Directory]::CreateDirectory($dest)|Out-Null
            & $adb -s $serial shell ('sh '+$board+'/run_board.sh '+$board+' '+$count) 2>&1 | Set-Content -Encoding utf8NoBOM (Join-Path $dest 'adb-run.log')
            $rcText=(Adb @('shell',('cat '+$board+'/results/exit-code.txt'))|Out-String).Trim()
            foreach ($name in @('board.log','metrics.json','init-memory.json','exit-code.txt','meminfo-before.txt','meminfo-after.txt','dma-before.txt','dma-after.txt','cpu-khz.txt','bootargs.txt','npu-frequency.txt','logits.f32','states.f32')) {
                $exists=(Adb @('shell',('if [ -f '+$board+'/results/'+$name+' ]; then echo EXISTS; fi'))|Out-String).Trim()
                if ($exists -eq 'EXISTS') { Adb @('pull',($board+'/results/'+$name),(Join-Path $dest $name))|Out-Null }
            }
            if ($rcText -notmatch '^\d+$' -or [int]$rcText -ne 0) { throw ('Board test failed '+$variant+' rc='+$rcText+'; evidence at '+$dest) }
            Write-Output ('BOARD_TEST_COMPLETE '+$variant+' steps='+$count)
            if ($Steps -eq 32) { break }
        }
        if ($variant -eq 'qat-qdq') {
            $dest=Join-Path $results ($variant+'/capture');[IO.Directory]::CreateDirectory($dest)|Out-Null
            & $adb -s $serial shell ('cd '+$board+' && { ./delta-dump > results/capture.log 2>&1; echo $? > results/capture-exit-code.txt; }')|Out-Null
            foreach ($name in @('capture.log','capture-exit-code.txt','input-feeds.f32','output-feeds.f32','feed-positions.i32')) {
                Adb @('pull',($board+'/results/'+$name),(Join-Path $dest $name))|Out-Null
            }
            $rcText=(Get-Content -Raw (Join-Path $dest 'capture-exit-code.txt')).Trim()
            if ($rcText -ne '0') { throw ('QAT native-input capture failed rc='+$rcText) }
            Adb @('shell','cat /proc/rk_dma_heap/alloc_bitmap')|Set-Content (Join-Path $dest 'dma-after.txt')
        }
    }
} finally {
    if ($restore) {
        if ($null -ne $webBefore) { & (Join-Path $project 'pico-chat/start.ps1') }
        else { Adb @('shell','/userdata/pico-chat/S98picochat start')|Out-Null }
        $after=$null
        for ($attempt=0;$attempt -lt 30;$attempt++) {
            try { $after=Health;if ($after.ready) {break} } catch {}
            Start-Sleep -Milliseconds 200
        }
        $after|ConvertTo-Json -Depth 7|Set-Content -Encoding utf8NoBOM (Join-Path $results 'restored-service.json')
        if (-not $after.ready -or $after.model -ne $before.model -or $after.checkpoint_sha256 -ne $before.checkpoint_sha256 -or $after.context_tokens -ne $before.context_tokens) { throw 'Original daemon was not restored with the original model/context' }
        Adb @('shell','cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /proc/rk_dma_heap/alloc_bitmap')|Set-Content (Join-Path $results 'restored-memory-governor.txt')
        Write-Output 'ORIGINAL_BOARD_SERVICE_RESTORED'
    }
}
