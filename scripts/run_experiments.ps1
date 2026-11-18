param(
    [int]$N = 256
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Bin = Join-Path $RepoRoot "cachesim.exe"
if (-not (Test-Path $Bin)) {
    throw "cachesim.exe not found at $Bin. Build it first with g++/clang++."
}

Write-Output "| workload | tile | pad | misses | miss rate % | compulsory | capacity | conflict | writebacks | AMAT |"
Write-Output "|---|---|---|---|---|---|---|---|---|---|"

function Invoke-Row([string[]]$SimArgs) {
    $line = & $Bin --csv --n $N @SimArgs
    $cols = $line.Split(",")
    $tileCol = if ($cols[0] -eq "tiled") { $cols[2] } else { "-" }
    Write-Output "| $($cols[0]) | $tileCol | $($cols[3]) | $($cols[4]) | $($cols[5]) | $($cols[6]) | $($cols[7]) | $($cols[8]) | $($cols[9]) | $($cols[10]) |"
}

Invoke-Row @("--workload", "naive")
Invoke-Row @("--workload", "naive", "--pad", "8")
Invoke-Row @("--workload", "ikj")
foreach ($T in 8, 16, 32, 64) {
    Invoke-Row @("--workload", "tiled", "--tile", "$T")
}
Invoke-Row @("--workload", "tiled", "--tile", "32", "--pad", "8")
