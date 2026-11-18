param(
    [int]$N = 256
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Bin = Join-Path $RepoRoot "cachesim.exe"
if (-not (Test-Path $Bin)) {
    throw "cachesim.exe not found at $Bin. Build it first with g++/clang++."
}

Write-Output "| bytes/elem | workload | tile | pad | miss rate % | misses | AMAT |"
Write-Output "|---|---|---|---|---|---|---|"

$Widths = @(
    @{ Elem = "8";   Bits = "64" },
    @{ Elem = "4";   Bits = "32" },
    @{ Elem = "2";   Bits = "16" },
    @{ Elem = "1";   Bits = "8" },
    @{ Elem = "0.5"; Bits = "4" }
)

$Configs = @(
    @("--workload", "naive", "--pad", "8"),
    @("--workload", "tiled", "--tile", "32", "--pad", "8"),
    @("--workload", "tiled", "--tile", "64", "--pad", "8")
)

foreach ($w in $Widths) {
    foreach ($cfg in $Configs) {
        $line = & $Bin --csv --n $N --bits $w.Bits @cfg
        $cols = $line.Split(",")
        $tileCol = if ($cols[0] -eq "tiled") { $cols[2] } else { "-" }
        Write-Output "| $($w.Elem) | $($cols[0]) | $tileCol | $($cols[3]) | $($cols[5]) | $($cols[4]) | $($cols[10]) |"
    }
}
