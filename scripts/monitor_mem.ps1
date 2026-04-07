while ($true) {
    $procs = Get-Process julia -ErrorAction SilentlyContinue
    if ($procs) {
        foreach ($p in $procs) {
            $ws = [math]::Round($p.WorkingSet64 / 1MB)
            $pm = [math]::Round($p.PrivateMemorySize64 / 1MB)
            $t = Get-Date -Format 'HH:mm:ss'
            Write-Host "$t | PID=$($p.Id) | WS=${ws}MB | Private=${pm}MB"
        }
    } else {
        $t = Get-Date -Format 'HH:mm:ss'
        Write-Host "$t | julia process not found"
    }
    Start-Sleep 10
}
