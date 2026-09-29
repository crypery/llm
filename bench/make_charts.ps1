# Bar-chart generator for bench results (no external deps, System.Drawing only).
#
#   powershell -ExecutionPolicy Bypass -File bench\make_charts.ps1
#
# Reads bench/bench_results.csv + bench/final_results.csv, writes PNGs to bench/charts/.
$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$outDir = Join-Path $scriptDir "charts"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
Add-Type -AssemblyName System.Drawing

function Read-CsvInv([string]$path) {
    $rows = @()
    foreach ($line in (Get-Content $path)) {
        if ($line -match '^\s*model,') { continue }
        if ($line.Trim() -eq '') { continue }
        $rows += ,($line -split ',')
    }
    return $rows
}

function New-GroupedChart([string]$title, [string]$ylabel, [string[]]$groups,
                          [string[]]$series, [double[][]]$values, [string]$outfile,
                          [double]$ymaxArg = -1) {
    $W = 940; $H = 540
    $ml = 70; $mr = 20; $mt = 56; $mb = 96
    $bmp = New-Object System.Drawing.Bitmap($W, $H)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.Clear([System.Drawing.Color]::White)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $fTitle = New-Object System.Drawing.Font("Arial", 15, [System.Drawing.FontStyle]::Bold)
    $fAxis  = New-Object System.Drawing.Font("Arial", 10)
    $fVal   = New-Object System.Drawing.Font("Arial", 9)
    $fLeg   = New-Object System.Drawing.Font("Arial", 11)
    $brText = [System.Drawing.Brushes]::Black
    $penGrid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(220,220,220), 1)
    $penAxis = New-Object System.Drawing.Pen([System.Drawing.Color]::Black, 1)
    $palette = @(
        [System.Drawing.Color]::FromArgb(46,134,222),
        [System.Drawing.Color]::FromArgb(229,80,80),
        [System.Drawing.Color]::FromArgb(39,174,96),
        [System.Drawing.Color]::FromArgb(243,156,18)
    )
    $sz = $g.MeasureString($title, $fTitle)
    $g.DrawString($title, $fTitle, $brText, ($W - $sz.Width) / 2, 10)

    $ymax = 0
    foreach ($s in $values) { foreach ($v in $s) { if ($v -gt $ymax) { $ymax = $v } } }
    if ($ymaxArg -gt 0) { $ymax = $ymaxArg }
    if ($ymax -le 0) { $ymax = 1 }
    $ymax = $ymax * 1.18

    $plotW = $W - $ml - $mr
    $plotH = $H - $mt - $mb
    $n = $groups.Count
    $ns = $series.Count
    $gw = $plotW / $n
    $bw = [Math]::Min(64, ($gw * 0.7) / $ns)

    # gridlines + y labels (6 steps)
    for ($k = 0; $k -le 6; $k++) {
        $yv = $ymax * $k / 6
        $y = $mt + $plotH - ($plotH * $k / 6)
        $g.DrawLine($penGrid, $ml, $y, $W - $mr, $y)
        $lab = "{0:0.#}" -f $yv
        $lsz = $g.MeasureString($lab, $fAxis)
        $g.DrawString($lab, $fAxis, $brText, $ml - $lsz.Width - 6, $y - 8)
    }
    $g.DrawString($ylabel, $fAxis, $brText, 8, $mt - 24)

    for ($i = 0; $i -lt $n; $i++) {
        $gx = $ml + $gw * $i
        for ($s = 0; $s -lt $ns; $s++) {
            $v = $values[$s][$i]
            $bh = [Math]::Max(2, $plotH * $v / $ymax)
            $bx = $gx + ($gw - $bw * $ns) / 2 + $bw * $s
            $by = $mt + $plotH - $bh
            $br = New-Object System.Drawing.SolidBrush($palette[$s % $palette.Count])
            $g.FillRectangle($br, $bx, $by, $bw, $bh)
            $g.DrawRectangle($penAxis, $bx, $by, $bw, [Math]::Max(1, $bh))
            $vl = "{0:0.#}" -f $v
            $vsz = $g.MeasureString($vl, $fVal)
            $g.DrawString($vl, $fVal, $brText, $bx + ($bw - $vsz.Width) / 2, $by - 16)
            $br.Dispose()
        }
        $gl = $groups[$i]
        $gsz = $g.MeasureString($gl, $fAxis)
        $g.DrawString($gl, $fAxis, $brText, $gx + ($gw - $gsz.Width) / 2, $mt + $plotH + 8)
    }
    $g.DrawLine($penAxis, $ml, $mt, $ml, $mt + $plotH)
    $g.DrawLine($penAxis, $ml, $mt + $plotH, $W - $mr, $mt + $plotH)

    # legend
    $lx = $ml
    for ($s = 0; $s -lt $ns; $s++) {
        $br = New-Object System.Drawing.SolidBrush($palette[$s % $palette.Count])
        $ly = $H - 34
        $g.FillRectangle($br, $lx, $ly, 18, 13)
        $g.DrawRectangle($penAxis, $lx, $ly, 18, 13)
        $g.DrawString($series[$s], $fLeg, $brText, $lx + 24, $ly - 3)
        $lsz = $g.MeasureString($series[$s], $fLeg)
        $lx += 24 + $lsz.Width + 30
        $br.Dispose()
    }

    $fTitle.Dispose(); $fAxis.Dispose(); $fVal.Dispose(); $fLeg.Dispose()
    $penGrid.Dispose(); $penAxis.Dispose()
    $g.Dispose()
    $bmp.Save($outfile, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Write-Host ("chart written: {0}" -f $outfile)
}

# --- bench_results.csv: CPU vs GPU speed ---
$bench = Join-Path $scriptDir "bench_results.csv"
if (Test-Path $bench) {
    $rows = Read-CsvInv $bench
    $groups = @(); $cpuG = @(); $gpuG = @(); $cpuP = @(); $gpuP = @()
    foreach ($r in $rows) {
        if ($r[1] -eq "cpu") {
            $groups += $r[0]
            $cpuG += [double]$r[4]; $cpuP += [double]$r[3]
        }
    }
    foreach ($gr in $groups) {
        foreach ($r in $rows) {
            if ($r[0] -eq $gr -and $r[1] -eq "gpu") {
                $gpuG += [double]$r[4]; $gpuP += [double]$r[3]
            }
        }
    }
    New-GroupedChart "Generation speed by model (tok/s, higher is better)" "tok/s" $groups @("CPU", "GPU") @($cpuG, $gpuG) (Join-Path $outDir "chart_gen.png")
    New-GroupedChart "Prompt processing speed by model (tok/s, higher is better)" "tok/s" $groups @("CPU", "GPU") @($cpuP, $gpuP) (Join-Path $outDir "chart_prompt.png")
} else {
    Write-Host "SKIP charts: missing $bench"
}

# --- final_results.csv: CPU vs GPU + answer match ---
$final = Join-Path $scriptDir "final_results.csv"
if (Test-Path $final) {
    $rows = Read-CsvInv $final
    $groups = @(); $cpuG = @(); $gpuG = @(); $match = @()
    foreach ($r in $rows) {
        $groups += $r[0]
        $cpuG += [double]$r[1]; $gpuG += [double]$r[4]; $match += [double]$r[8]
    }
    New-GroupedChart "Final test: generation speed CPU vs GPU (tok/s)" "tok/s" $groups @("CPU", "GPU") @($cpuG, $gpuG) (Join-Path $outDir "chart_final_gen.png")
    # NOTE: single-series values must be wrapped explicitly: @($match) gets
    # unwrapped by PowerShell when binding to double[][], zeroing all but [0]
    $matchVals = New-Object 'double[][]' 1
    $matchVals[0] = [double[]]$match
    New-GroupedChart "Final test: CPU/GPU answer match, common prefix (%)" "%" $groups @("match") $matchVals (Join-Path $outDir "chart_final_match.png") 100
} else {
    Write-Host "SKIP charts: missing $final"
}
