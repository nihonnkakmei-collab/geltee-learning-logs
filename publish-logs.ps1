param([int]$Step = 0, [switch]$Initial)
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$git = 'git'
$message = if ($Initial) { 'Initialize Geltee learning logs' } else { "Update Geltee learning log at step $Step" }
& $git -C $repo add README.md INITIAL_INSTRUCTION.txt .gitignore adaptive_train_step.py run-infinite.ps1 publish-logs.ps1 curate_dataset.py source_allowlist.example.json curation_instructions.md logs
& $git -C $repo diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    & $git -C $repo commit -m $message
    if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
    & $git -C $repo push -u origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw 'git push failed' }
}
