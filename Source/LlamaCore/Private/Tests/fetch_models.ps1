# Copyright 2025-current Getnamo.
#
# Downloads embedding models used by the LlamaCore RAG automation tests.
# Default target: bge-small-en-v1.5 Q4_K_M (~33 MB, dim 384).
#
# Usage (from any directory):
#   pwsh -ExecutionPolicy Bypass -File fetch_models.ps1
#   pwsh -ExecutionPolicy Bypass -File fetch_models.ps1 -All
#
# Models are placed under <project>/Saved/Models/ — the same root used by
# FLLMModelParams.PathToModel when prefixed with `./`.

param(
    [switch]$All,
    [string]$DestRoot
)

$ErrorActionPreference = 'Stop'

if (-not $DestRoot) {
    # Walk up from this script: Plugins/Llama-Unreal/Source/LlamaCore/Private/Tests/ -> project root
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ProjectRoot = (Resolve-Path (Join-Path $ScriptDir '..\..\..\..\..\..\..')).Path
    $DestRoot = Join-Path $ProjectRoot 'Saved\Models'
}

New-Item -ItemType Directory -Force -Path $DestRoot | Out-Null

$Models = @(
    @{
        Name = 'bge-small-en-v1.5-q4_k_m.gguf'
        Url  = 'https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q4_k_m.gguf'
        Size = 33MB
        Required = $true
    }
)

if ($All) {
    $Models += @(
        @{
            Name = 'nomic-embed-text-v1.5.Q4_K_M.gguf'
            Url  = 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf'
            Size = 85MB
            Required = $false
        }
    )
}

foreach ($M in $Models) {
    $Dest = Join-Path $DestRoot $M.Name
    if (Test-Path $Dest) {
        Write-Host "[skip] $($M.Name) already at $Dest"
        continue
    }
    Write-Host "[get ] $($M.Name) -> $Dest"
    try {
        Invoke-WebRequest -Uri $M.Url -OutFile $Dest -UseBasicParsing
    } catch {
        if ($M.Required) {
            Write-Error "Failed to download required model $($M.Name): $($_.Exception.Message)"
            throw
        }
        Write-Warning "Optional model $($M.Name) skipped: $($_.Exception.Message)"
    }
}

Write-Host ""
Write-Host "Models ready at: $DestRoot"
Write-Host "In FLLMModelParams.PathToModel use './<filename>' to reference them."
