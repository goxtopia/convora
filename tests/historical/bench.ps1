param([string]$Variant='saturating-conv',[string]$BoardName='conv-saturating-conv',[int]$Steps=32)
$ErrorActionPreference='Stop'
$adb='D:\platform-tools\adb.exe';$serial='b9deb5cb2fb52c4f'
$board='/mnt/sdcard/pico-models/qat-toolchain-match-20261029/'+$BoardName
$original='/mnt/sdcard/pico-models/delta-sft-qat-20261029/qat-qdq'
function A([string[]]$ArgList){$r=& $adb -s $serial @ArgList;if($LASTEXITCODE){throw ('ADB failed '+($ArgList -join ' '))};return $r}
$dest=Join-Path $PSScriptRoot ('board-closed-'+$Variant)
if(Test-Path $dest){throw 'Existing evidence'}
[IO.Directory]::CreateDirectory($dest)|Out-Null
$dma=A @('shell','cat /proc/rk_dma_heap/alloc_bitmap')|Out-String
if($dma -notmatch 'Used:\s+12 KiB'){throw 'NPU not idle'}
A @('shell',('cd '+$board+' && mkdir models data results && ln -s ../model.rknn models/tcn_engram.rknn && ln -s '+$original+'/delta-bench delta-bench && ln -s '+$original+'/run_board.sh run_board.sh'))|Out-Null
foreach($name in @('embedding.f32','memory.i8','memory_scale.f32','sequence.i32','reference.f32','state-reference.f32')){A @('shell',('ln -s '+$original+'/data/'+$name+' '+$board+'/data/'+$name))|Out-Null}
A @('shell',('sh '+$board+'/run_board.sh '+$board+' '+$Steps))|Set-Content (Join-Path $dest 'command.log')
foreach($name in @('metrics.json','board.log','exit-code.txt','logits.f32','states.f32','dma-before.txt','dma-after.txt','meminfo-after.txt')){A @('pull',($board+'/results/'+$name),(Join-Path $dest $name))|Out-Null}
A @('shell','cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /proc/rk_dma_heap/alloc_bitmap; pidof pico-chatd')|Set-Content (Join-Path $dest 'final-board.txt')
