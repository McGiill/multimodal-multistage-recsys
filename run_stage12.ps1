param(
    [ValidateSet('demo', 'verify', 'train', 'evaluate', 'serve')]
    [string]$Action = 'demo'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $taskWorkspace = Split-Path -Parent $PSScriptRoot
    $taskPackages = Join-Path $taskWorkspace 'work/integration_packages'
    $taskMlPackages = Join-Path $taskWorkspace 'work/packages'
    $taskBundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if ((Test-Path -LiteralPath $taskBundledPython) -and (Test-Path -LiteralPath $taskPackages) -and (Test-Path -LiteralPath $taskMlPackages)) {
        $taskPython = $taskBundledPython
        $env:PYTHONPATH = "$taskPackages;$taskMlPackages"
    } else {
        $taskPython = (Get-Command python -ErrorAction Stop).Source
    }
    $env:PRELOAD_MODELS = '0'
    if ($Action -eq 'serve') {
        & $taskPython -m uvicorn src.serving.main:app --host 127.0.0.1 --port 8000
    } else {
        & $taskPython scripts/stage12_team.py --action $Action
    }
    if ($LASTEXITCODE -ne 0) { throw "Stage 1/2 command failed (exit $LASTEXITCODE)" }
} finally {
    Pop-Location
}
