$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$python = 'C:\Users\matsu\AppData\Local\Programs\Python\Python312\python.exe'
$state = 'C:\Users\matsu\Documents\Codex\2026-08-08\rru\work\geltee-infinite-state'
$logs = Join-Path $repo 'logs'
$latestJson = Join-Path $logs 'latest.json'
$latestMd = Join-Path $logs 'latest.md'
$counter = Join-Path $state 'step.txt'
New-Item -ItemType Directory -Force -Path $state, $logs | Out-Null

if (-not (Test-Path -LiteralPath $counter)) { Set-Content -LiteralPath $counter -Value '0' -Encoding ASCII }
& (Join-Path $repo 'publish-logs.ps1') -Initial

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
- Baseline gate: $($result.baseline.score)/$($result.baseline.total)
- Candidate gate: $($result.candidate.score)/$($result.candidate.total)
- Promoted: $($result.promoted)
- Learning rate: $($result.train.lr)
- Mean loss: $($result.train.mean_loss)
- GPT-1 exceeded: not yet established

The GPT-1 statement requires a shared benchmark against a reproduced baseline; the Geltee gate alone is not used as proof.
"@
    Set-Content -LiteralPath $latestMd -Value $md -Encoding UTF8
    if (($step % 10) -eq 0) { & (Join-Path $repo 'publish-logs.ps1') -Step $step }
}

