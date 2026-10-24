param([Parameter(Mandatory=$true)][string]$Destination)
$ErrorActionPreference = 'Stop'
$Candidate = Split-Path $PSScriptRoot -Parent
$Destination = (Resolve-Path -LiteralPath $Destination).Path
if (-not (Test-Path -LiteralPath (Join-Path $Destination '.git'))) {
    throw 'Destination must be an existing clone; do not reinitialize a repository.'
}
$State = @(git -C "$Destination" status --porcelain=v1)
if ($LASTEXITCODE -ne 0 -or $State.Count -ne 0) {
    throw 'Target repository is unavailable or dirty; preserve its changes and pause.'
}
$Manifest = Get-Content -LiteralPath (Join-Path $Candidate 'PUBLIC_MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$Names = @($Manifest.files.PSObject.Properties.Name) + @('PUBLIC_MANIFEST.json')
foreach ($Name in $Names) {
    $Source = Join-Path $Candidate $Name
    if ($Name -ne 'PUBLIC_MANIFEST.json') {
        $Expected = $Manifest.files.PSObject.Properties[$Name].Value
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash -ne $Expected) {
            throw "Candidate hash mismatch: $Name"
        }
    }
    $Target = Join-Path $Destination $Name
    if (Test-Path -LiteralPath $Target) {
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash -ne (Get-FileHash -Algorithm SHA256 -LiteralPath $Target).Hash) {
            throw "Existing target differs; merge manually before continuing: $Name"
        }
    }
}
foreach ($Name in $Names) {
    $Target = Join-Path $Destination $Name
    if (-not (Test-Path -LiteralPath $Target)) {
        New-Item -ItemType Directory -Path (Split-Path $Target -Parent) -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $Candidate $Name) -Destination $Target
    }
}
git -C "$Destination" add -- $Names
if ($LASTEXITCODE -ne 0) { throw 'git add failed; preserve files and inspect.' }
# Preserve reviewed upstream/overlay bytes and unified-diff context verbatim.
# Whitespace policy applies to newly authored packaging, not inherited source.
git -C "$Destination" diff --cached --check -- . ':(exclude)overlay/**' ':(exclude)patches/**'
if ($LASTEXITCODE -ne 0) { throw 'Whitespace check failed; review before committing.' }
Write-Host 'Files staged. No commit or push executed. Review git diff --cached --stat first.'
