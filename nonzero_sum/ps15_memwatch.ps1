# Memory watchdog for the p sweep (sgb15_psweep.jl). Stops ONLY our julia (matched by command line) when free RAM < $KillGB.
# Other sessions' processes are never touched. Exits when our julia ends or after killing it.
param([double]$KillGB = 3.0, [double]$WarnGB = 6.0, [int]$Every = 10)
$log = Join-Path $PSScriptRoot 'logs\sgb15_fac\ps15\memwatch.log'
function Say($m) { $line = (Get-Date -Format 'MM-dd HH:mm:ss') + ' ' + $m; Add-Content -Path $log -Value $line; [Console]::Out.WriteLine($line); [Console]::Out.Flush() }
$seen = $false; $warned = $false; $minFree = 1e9
Say ("start: kill < $KillGB GB, warn < $WarnGB GB, every $Every s")
while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $free = $os.FreePhysicalMemory / 1MB
    if ($free -lt $minFree) { $minFree = $free }
    # Git bash launches julia without visible args, so match julia.exe whose parent bash command line names the script
    $procs = @(Get-CimInstance Win32_Process)
    $par = @{}; foreach ($q in $procs) { $par[[int]$q.ProcessId] = $q.CommandLine }
    $p = @($procs | Where-Object { $_.Name -eq 'julia.exe' -and (($_.CommandLine -like '*sgb15_psweep.jl*') -or ($par[[int]$_.ParentProcessId] -like '*sgb15_psweep.jl*')) })
    if ($p.Count -gt 0) { $seen = $true }
    elseif ($seen) { Say ("our julia ended; min free {0:N1} GB; watchdog exits" -f $minFree); exit 0 }
    if ($p.Count -gt 0 -and $free -lt $KillGB) {
        foreach ($q in $p) { Stop-Process -Id $q.ProcessId -Force -Confirm:$false -ErrorAction Stop }
        Say ("KILLED our julia {0} (ws {1:N1} GB): free {2:N1} GB < {3} GB" -f (($p | ForEach-Object { $_.ProcessId }) -join ','), (($p | Measure-Object WorkingSetSize -Sum).Sum / 1GB), $free, $KillGB)
        exit 2
    }
    if ($free -lt $WarnGB -and -not $warned) {
        Say ("WARN free {0:N1} GB < {1} GB (our julia ws {2:N1} GB)" -f $free, $WarnGB, (($p | Measure-Object WorkingSetSize -Sum).Sum / 1GB)); $warned = $true
    } elseif ($free -ge $WarnGB + 1) { $warned = $false }
    Start-Sleep -Seconds $Every
}
