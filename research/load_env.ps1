# Dot-source from the repo root:  . .\research\load_env.ps1
# Reads .env.local and sets every non-empty KEY=VALUE as an environment variable of THIS
# PowerShell window only. Prints variable names, never values.
param([string]$Path = (Join-Path (Split-Path $PSScriptRoot -Parent) '.env.local'))

$root = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path $Path)) {
    Write-Error "Not found: $Path (copy .env.example to .env.local first)"
    return
}
$loaded = @()
foreach ($raw in Get-Content -Encoding UTF8 $Path) {
    $line = $raw.Trim()
    if ($line -eq '' -or $line.StartsWith('#')) { continue }
    $i = $line.IndexOf('=')
    if ($i -lt 1) { continue }
    $key = $line.Substring(0, $i).Trim()
    $val = $line.Substring($i + 1).Trim().Trim('"').Trim("'")
    if ($val -eq '') { continue }
    if ($key -eq 'MOSEKLM_LICENSE_FILE' -and -not [System.IO.Path]::IsPathRooted($val)) {
        $val = Join-Path $root $val
    }
    Set-Item -Path "Env:$key" -Value $val
    $loaded += $key
}
Write-Host ('Loaded: ' + ($loaded -join ', '))
if ($env:STONEPACKING_ALLOW_PAID_API -eq '1') {
    Write-Warning 'Paid API gate is ON in this window (STONEPACKING_ALLOW_PAID_API=1).'
}
