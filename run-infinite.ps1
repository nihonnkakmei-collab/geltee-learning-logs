$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$python = $env:GELTEE_PYTHON
$state = $env:GELTEE_STATE_DIR
$env:PYTHONPATH = $env:GELTEE_PYTHONPATH
if (-not $python -or -not $state -or -not $env:PYTHONPATH) { throw 'Missing local Geltee launcher configuration.' }
$logs = Join-Path $repo 'logs'
$latestJson = Join-Path $logs 'latest.json'
$latestMd = Join-Path $logs 'latest.md'
$researchDir = Join-Path $repo 'research'
$researchLatest = Join-Path $researchDir 'latest.json'
$curatedDir = Join-Path $state 'curated'
$env:GELTEE_CURATED_DATA = Join-Path $curatedDir 'training_candidates.jsonl'
$counter = Join-Path $state 'step.txt'
New-Item -ItemType Directory -Force -Path $state, $logs, $researchDir, $curatedDir | Out-Null

if (-not (Test-Path -LiteralPath $counter)) { Set-Content -LiteralPath $counter -Value '0' -Encoding ASCII }
& (Join-Path $repo 'publish-logs.ps1') -Initial
# Populate the local curated queue before the first candidate step, so Gelqeen
# remains useful while Codex is unavailable. Failures leave the guarded loop
# running on the original local corpus.
& $python (Join-Path $repo 'auto_intake.py') --policy (Join-Path $repo 'research_policy.json') --out-dir $curatedDir
if ($LASTEXITCODE -ne 0) { Write-Warning 'Initial automatic data intake failed; continuing with the guarded local-only trainer.' }

while (-not (Test-Path -LiteralPath (Join-Path $repo 'STOP'))) {
    $step = [int](Get-Content -LiteralPath $counter -Raw) + 1
    & $python (Join-Path $repo 'adaptive_train_step.py') --step $step --state-dir $state --result $latestJson
    if ($LASTEXITCODE -ne 0) { throw "training step $step failed" }
    Set-Content -LiteralPath $counter -Value $step -Encoding ASCII
    $result = Get-Content -LiteralPath $latestJson -Raw -Encoding UTF8 | ConvertFrom-Json
    $md = @"
# Latest Geltee learning status

- Step: $step
- Updated: $($result.timestamp)
- Baseline gate: $($result.baseline.gate.score)/$($result.baseline.gate.total)
- Candidate gate: $($result.candidate.gate.score)/$($result.candidate.gate.total)
- Baseline holdout NLL: $($result.baseline.holdout_nll)
- Candidate holdout NLL: $($result.candidate.holdout_nll)
- Decision: $($result.decision)
- Learning rate: $($result.train.lr)
- Mean loss: $($result.train.mean_loss)
- GPT-1 exceeded: not yet established

The fixed 100-case gate is never used for training. A candidate is promoted only when it preserves gate performance and improves an independent holdout set.
"@
    Set-Content -LiteralPath $latestMd -Value $md -Encoding UTF8
    if (($step % 10) -eq 0) {
        # Gelqeen discovery is metadata-only. It cannot download or train on web data.
        & $python (Join-Path $repo 'research_sources.py') --status $latestJson --policy (Join-Path $repo 'research_policy.json') --result $researchLatest
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Research discovery failed; continuing with the guarded local-only trainer.' }
        & $python (Join-Path $repo 'auto_intake.py') --policy (Join-Path $repo 'research_policy.json') --out-dir $curatedDir
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Automatic data intake failed; continuing with the guarded local-only trainer.' }
        & (Join-Path $repo 'publish-logs.ps1') -Step $step
    }
}
