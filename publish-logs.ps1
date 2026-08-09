param([int]$Step = 0, [switch]$Initial)
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$git = 'git'
$message = if ($Initial) { 'Initialize Geltee learning logs' } else { "Update Geltee learning log at step $Step" }
& $git -C $repo add README.md INITIAL_INSTRUCTION.txt .gitignore adaptive_train_step.py run-infinite.ps1 publish-logs.ps1 logs
if (-not (& $git -C $repo diff --cached --quiet)) {
    & $git -C $repo commit -m $message
    & $git -C $repo push -u origin HEAD:main
}

