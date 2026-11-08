# Build one of the six batches of the factorial study. Run it again to resume: finished walls are skipped.
# Usage, from the repository root: powershell -File research\walls\factorial\run_batch.ps1 -Batch 1 -Python <python.exe>
param(
    [Parameter(Mandatory = $true)][ValidateRange(1, 6)][int]$Batch,
    [string]$Python = 'python',
    [switch]$DryRun
)

$Batches = @{
    1 = 's313-TH_T20', 's313-TH_T80', 's313-AL_PR', 's313-AL_IR'
    2 = 's484-TH_T20', 's484-TH_T80', 's484-AL_R', 's484-AL_IR'
    3 = 's717-TH_T20', 's717-TH_T80', 's717-AL_PR', 's717-AL_IR'
    4 = 's720-TH_T20', 's720-TH_T80', 's720-AL_R', 's720-AL_IR'
    5 = 's761-TH_T20', 's761-TH_T80', 's761-AL_PR', 's761-AL_IR'
    6 = 's313-AL_R', 's484-AL_PR', 's717-AL_R', 's720-AL_PR', 's761-AL_R'
}

$Extra = @()
if ($DryRun) { $Extra = @('--dry-run') }

foreach ($Block in $Batches[$Batch]) {
    $Name = "factorial-$Block"
    & $Python -B -m research.runner.batch plan "research\walls\factorial\$Name.yaml"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B -m research.runner.batch run $Name @Extra
    # 1 only means some walls failed; Ctrl+C (130) and TERMINATION_FAILED (2) stop the batch
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 1) { exit $LASTEXITCODE }
}
