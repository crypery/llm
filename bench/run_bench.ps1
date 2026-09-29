# Benchmark harness for the llm module across models and devices.
#
#   powershell -ExecutionPolicy Bypass -File bench\run_bench.ps1
#   powershell -ExecutionPolicy Bypass -File bench\run_bench.ps1 -Only gemma3,lfm2
#
# Requires test_llm_cpu.exe and test_llm_gpu.exe (build via build_llm.bat all).
# Missing models / binaries are skipped (row with zeros) instead of failing.
param(
    [string]$Threads = "4",
    [int]$Ctx = 4096,
    [int]$Predict = 96,
    [string]$Prompt = "Расскажи кратко, что такое искусственный интеллект и назови три его основных применения.",
    [string[]]$Only = @()
)

$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir

$CpuExe = Join-Path $root "test_llm_cpu.exe"
$GpuExe = Join-Path $root "test_llm_gpu.exe"
$OutCsv = Join-Path $scriptDir "bench_results.csv"

$models = @(
    @{ name = "gemma3";    file = Join-Path $root "gemma3.gguf" },
    @{ name = "gemma3-4b"; file = Join-Path $root "gemma3-4b.gguf" },
    @{ name = "lfm2";      file = Join-Path $root "lfm2.gguf" },
    @{ name = "lfm2-4b";   file = Join-Path $root "lfm2-4B.gguf" },
    @{ name = "qwen3";     file = Join-Path $root "qwen3.gguf" },
    @{ name = "qwen3-4b";  file = Join-Path $root "qwen3-4b.gguf" }
)

if ($Only.Count -gt 0) {
    # normalize: via "-File" the list may arrive as one comma-joined string
    $Only = @($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' })
    $models = @($models | Where-Object { $Only -contains $_.name })
}

function Invoke-Bench([string]$exe, [string]$device, [string]$modelName, [string]$modelFile) {
    $row = [pscustomobject]@{
        model = $modelName; device = $device; threads = $Threads
        prompt_tok_s = 0.0; gen_tok_s = 0.0; n_gen = 0; total_s = 0.0
    }
    if (-not (Test-Path $exe)) {
        Write-Host ("{0,-12} {1,-4} SKIP: missing binary {2}" -f $modelName, $device, (Split-Path -Leaf $exe))
        return $row
    }
    if (-not (Test-Path $modelFile)) {
        Write-Host ("{0,-12} {1,-4} SKIP: missing model {2}" -f $modelName, $device, (Split-Path -Leaf $modelFile))
        return $row
    }

    $env:CHAT_TIMING = "1"
    $env:LLM_NO_CACHE = "1"
    Remove-Item Env:LLM_NO_BATCH -ErrorAction SilentlyContinue
    Remove-Item Env:LLM_CPU_ONLY -ErrorAction SilentlyContinue

    $cli = @("-m", $modelFile, "--threads", "$Threads", "-c", "$Ctx", "--lang", "ru",
              "--predict", "$Predict", "--temp", "0.0", "-p", $Prompt)
    if ($device -eq "gpu") { $cli = @("--device", "gpu") + $cli }

    $errFile = Join-Path $scriptDir ("_err_" + $modelName + "_" + $device + ".txt")
    & $exe @cli 1> $null 2> $errFile
    $stderr = ""
    if (Test-Path $errFile) { $stderr = Get-Content -Raw $errFile; Remove-Item $errFile -ErrorAction SilentlyContinue }

    $promptTok = 0.0; $genTok = 0.0; $nGen = 0; $totalS = 0.0
    if ($stderr -match '\[llm\] prompt:\s+(\d+)\s+tok in\s+([\d.]+)\s+s\s*=\s*([\d.]+)\s+tok/s') {
        $promptTok = [double]$Matches[3]
    }
    if ($stderr -match '\[llm\] generate:\s+(\d+)\s+tok in\s+([\d.]+)\s+s\s*=\s*([\d.]+)\s+tok/s') {
        $genTok = [double]$Matches[3]
        $nGen = [int]$Matches[1]
    }
    if ($stderr -match '\[llm\] total wall time \(main\):\s+([\d.]+)\s+s') {
        $totalS = [double]$Matches[1]
    }
    $row.prompt_tok_s = [math]::Round($promptTok,1)
    $row.gen_tok_s    = [math]::Round($genTok,1)
    $row.n_gen        = $nGen
    $row.total_s      = [math]::Round($totalS,1)
    return $row
}

$rows = @()
foreach ($m in $models) {
    foreach ($dev in @("cpu","gpu")) {
        $exe = if ($dev -eq "gpu") { $GpuExe } else { $CpuExe }
        $r = Invoke-Bench $exe $dev $m.name $m.file
        $rows += $r
        Write-Host ("{0,-12} {1,-4} prompt={2,7} tok/s  gen={3,7} tok/s  (n_gen={4}, total={5}s)" -f $r.model,$r.device,$r.prompt_tok_s,$r.gen_tok_s,$r.n_gen,$r.total_s)
    }
}

# merge with previous results (chunked runs overwrite only their own rows)
$all = @()
if (Test-Path $OutCsv) {
    $prev = Import-Csv -Path $OutCsv
    $keys = @($rows | ForEach-Object { $_.model + "|" + $_.device })
    $all += @($prev | Where-Object { -not ($keys -contains ($_.model + "|" + $_.device)) })
}
$all += $rows
$rows = $all

# Invariant-culture CSV (dot decimals) so results parse on any locale.
$ci = [System.Globalization.CultureInfo]::InvariantCulture
$lines = @("model,device,threads,prompt_tok_s,gen_tok_s,n_gen,total_s")
foreach ($r in $rows) {
    $lines += ("{0},{1},{2},{3},{4},{5},{6}" -f $r.model, $r.device, $r.threads,
        $r.prompt_tok_s.ToString($ci), $r.gen_tok_s.ToString($ci), $r.n_gen,
        $r.total_s.ToString($ci))
}
[System.IO.File]::WriteAllLines($OutCsv, $lines, [System.Text.Encoding]::UTF8)
Write-Host ""
Write-Host "Results written to $OutCsv"
