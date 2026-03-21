# =============================================================================
# Part 2: Automated Evaluation Framework - End-to-End Pipeline
# Conversational AI Assignment 2 - Group 122
# =============================================================================
#
# Usage:
#   .\run_part2.ps1                  # Full pipeline (all steps)
#   .\run_part2.ps1 -SkipUrls        # Skip URL generation (use existing JSONs)
#   .\run_part2.ps1 -SkipQuestions   # Skip question generation (use existing Q&A)
#   .\run_part2.ps1 -NoDashboard     # Don't launch Streamlit dashboard at end
#   .\run_part2.ps1 -OnlyDashboard   # Only launch the dashboard
#
# Pipeline Steps:
#   0. Environment setup & dependency check
#   1. Generate Wikipedia URL datasets (generate_urls.py)
#   2. Generate 100 Q&A pairs (Part2_QuestionGen.py)
#   3. Run evaluation pipeline (Part2_Evaluation.py)
#      -> MRR, ROUGE-L, BERTScore
#      -> Ablation study (5 configurations)
#      -> LLM-as-Judge
#      -> Error analysis
#      -> HTML report generation
#   4. Launch evaluation dashboard (Part2_Dashboard.py)

param(
    [switch]$SkipUrls,
    [switch]$SkipQuestions,
    [switch]$NoDashboard,
    [switch]$OnlyDashboard
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Project directory
# ---------------------------------------------------------------------------
$PROJECT_DIR = "c:\Users\koush\OneDrive\Documents\BITS Pilani\Conversational AI\Assignment 2"
Set-Location $PROJECT_DIR

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

function Write-Step {
    param([string]$StepNum, [string]$Message)
    Write-Host "`n────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  Step $StepNum : $Message" -ForegroundColor Cyan
    Write-Host "────────────────────────────────────────" -ForegroundColor DarkGray
}

function Write-Ok {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "  [!!] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "  [FAIL] $Message" -ForegroundColor Red
}

function Test-FileRecent {
    param([string]$Path, [int]$MaxAgeMinutes = 60)
    if (-not (Test-Path $Path)) { return $false }
    $age = (Get-Date) - (Get-Item $Path).LastWriteTime
    return $age.TotalMinutes -le $MaxAgeMinutes
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  PART 2: Automated Evaluation Framework" -ForegroundColor Magenta
Write-Host "  Conversational AI Assignment 2 - Group 122" -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  Working dir : $PROJECT_DIR" -ForegroundColor Gray
Write-Host "  Started at  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# ---------------------------------------------------------------------------
# Jump to dashboard only
# ---------------------------------------------------------------------------
if ($OnlyDashboard) {
    Write-Step "4" "Launching Evaluation Dashboard"
    if (-not (Test-Path "evaluation_results.csv")) {
        Write-Warn "evaluation_results.csv not found - dashboard may show incomplete data"
    }
    Write-Host "  Starting Streamlit at http://localhost:8501" -ForegroundColor Green
    Write-Host "  Press Ctrl+C to stop`n" -ForegroundColor Gray
    streamlit run ConversationalAI_Assignment_2_Group_122_Part2_Dashboard.py
    exit 0
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 0: Environment & Dependencies
# ═══════════════════════════════════════════════════════════════════════════

Write-Step "0" "Checking environment and dependencies"

# Check Python
try {
    $pyVersion = python --version 2>&1
    Write-Ok "Python: $pyVersion"
} catch {
    Write-Err "Python not found. Please install Python 3.10+ and add it to PATH."
    exit 1
}

# Check key packages
$missingPackages = @()
$requiredPackages = @(
    @{ Module = "sentence_transformers"; Pip = "sentence-transformers" },
    @{ Module = "ctransformers";         Pip = "ctransformers" },
    @{ Module = "faiss";                 Pip = "faiss-cpu" },
    @{ Module = "rank_bm25";             Pip = "rank-bm25" },
    @{ Module = "rouge_score";           Pip = "rouge-score" },
    @{ Module = "bert_score";            Pip = "bert-score" },
    @{ Module = "pandas";                Pip = "pandas" },
    @{ Module = "matplotlib";            Pip = "matplotlib" },
    @{ Module = "streamlit";             Pip = "streamlit" },
    @{ Module = "tqdm";                  Pip = "tqdm" },
    @{ Module = "requests";              Pip = "requests" },
    @{ Module = "bs4";                   Pip = "beautifulsoup4" },
    @{ Module = "plotly";                Pip = "plotly" }
)

foreach ($pkg in $requiredPackages) {
    $check = python -c "import $($pkg.Module)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missingPackages += $pkg.Pip
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Warn "Missing packages: $($missingPackages -join ', ')"
    $install = Read-Host "  Install them now? (Y/n)"
    if ($install -ne "n" -and $install -ne "N") {
        $pipCmd = "pip install $($missingPackages -join ' ')"
        Write-Host "  Running: $pipCmd" -ForegroundColor Gray
        Invoke-Expression $pipCmd
        if ($LASTEXITCODE -ne 0) {
            Write-Err "pip install failed. Please install manually:"
            Write-Host "    pip install -r requirements.txt" -ForegroundColor Yellow
            exit 1
        }
        Write-Ok "All packages installed"
    } else {
        Write-Err "Cannot proceed without required packages."
        exit 1
    }
} else {
    Write-Ok "All required packages are installed"
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Generate URL Datasets
# ═══════════════════════════════════════════════════════════════════════════

if (-not $SkipUrls) {
    Write-Step "1" "Generating Wikipedia URL datasets"

    $fixedExists = Test-Path "fixed_url.json"
    $randomExists = Test-Path "random_url.json"

    if ($fixedExists -and $randomExists) {
        $fixed = Get-Content "fixed_url.json" | ConvertFrom-Json
        $random = Get-Content "random_url.json" | ConvertFrom-Json
        Write-Host "  Existing datasets found:" -ForegroundColor Gray
        Write-Host "    - fixed_url.json  : $($fixed.Count) entries" -ForegroundColor Gray
        Write-Host "    - random_url.json : $($random.Count) entries" -ForegroundColor Gray

        $regen = Read-Host "  Regenerate? (y/N)"
        if ($regen -ne "y" -and $regen -ne "Y") {
            Write-Ok "Using existing URL datasets"
            $SkipUrls = $true
        }
    }

    if (-not $SkipUrls) {
        Write-Host "  Fetching 200 fixed + 300 random Wikipedia articles..." -ForegroundColor Gray
        Write-Host "  This takes ~10-20 minutes (network-dependent)" -ForegroundColor Yellow

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        python generate_urls.py --fixed 200 --random 300 --group GROUP_122
        $sw.Stop()

        if ($LASTEXITCODE -ne 0) {
            Write-Err "generate_urls.py failed (exit code $LASTEXITCODE)"
            exit 1
        }

        # Validate outputs
        $valid = $true
        foreach ($f in @("fixed_url.json", "random_url.json")) {
            if (Test-Path $f) {
                $data = Get-Content $f | ConvertFrom-Json
                Write-Ok "$f : $($data.Count) entries"
            } else {
                Write-Err "$f was not created"
                $valid = $false
            }
        }
        if (-not $valid) { exit 1 }

        $elapsed = $sw.Elapsed.ToString("hh\:mm\:ss")
        Write-Ok "URL generation complete ($elapsed)"
    }
} else {
    Write-Step "1" "Skipping URL generation (--SkipUrls)"
    # Verify files exist
    foreach ($f in @("fixed_url.json", "random_url.json")) {
        if (Test-Path $f) {
            $data = Get-Content $f | ConvertFrom-Json
            Write-Ok "$f : $($data.Count) entries"
        } else {
            Write-Err "$f not found! Remove -SkipUrls to generate it."
            exit 1
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Generate Evaluation Questions (Section 2.1)
# ═══════════════════════════════════════════════════════════════════════════

if (-not $SkipQuestions) {
    Write-Step "2" "Generating 100 Q&A pairs (Section 2.1)"

    if (Test-Path "evaluation_questions.json") {
        $questions = Get-Content "evaluation_questions.json" | ConvertFrom-Json
        Write-Host "  Existing evaluation_questions.json found: $($questions.Count) questions" -ForegroundColor Gray

        $regen = Read-Host "  Regenerate? (y/N)"
        if ($regen -ne "y" -and $regen -ne "Y") {
            Write-Ok "Using existing Q&A pairs"
            $SkipQuestions = $true
        }
    }

    if (-not $SkipQuestions) {
        Write-Host "  Using TinyLlama to generate questions from corpus..." -ForegroundColor Gray
        Write-Host "  Distribution: 40 factual, 20 comparative, 25 inferential, 15 multi-hop" -ForegroundColor Gray
        Write-Host "  This may take 15-30 minutes (LLM inference)" -ForegroundColor Yellow

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        python ConversationalAI_Assignment_2_Group_122_Part2_QuestionGen.py
        $sw.Stop()

        if ($LASTEXITCODE -ne 0) {
            Write-Err "QuestionGen.py failed (exit code $LASTEXITCODE)"
            exit 1
        }

        if (Test-Path "evaluation_questions.json") {
            $questions = Get-Content "evaluation_questions.json" | ConvertFrom-Json
            Write-Ok "evaluation_questions.json : $($questions.Count) questions"

            # Show category breakdown
            $categories = $questions | Group-Object -Property category
            foreach ($cat in $categories) {
                Write-Host "    - $($cat.Name): $($cat.Count)" -ForegroundColor Gray
            }
        } else {
            Write-Err "evaluation_questions.json was not created"
            exit 1
        }

        $elapsed = $sw.Elapsed.ToString("hh\:mm\:ss")
        Write-Ok "Question generation complete ($elapsed)"
    }
} else {
    Write-Step "2" "Skipping question generation (--SkipQuestions)"
    if (Test-Path "evaluation_questions.json") {
        $questions = Get-Content "evaluation_questions.json" | ConvertFrom-Json
        Write-Ok "evaluation_questions.json : $($questions.Count) questions"
    } else {
        Write-Err "evaluation_questions.json not found! Remove -SkipQuestions to generate it."
        exit 1
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Run Evaluation Pipeline (Sections 2.2 - 2.5)
# ═══════════════════════════════════════════════════════════════════════════

Write-Step "3" "Running evaluation pipeline (Sections 2.2-2.5)"
Write-Host "  Sub-steps:" -ForegroundColor Gray
Write-Host "    3a. RAG inference on 100 questions" -ForegroundColor Gray
Write-Host "    3b. Compute MRR, ROUGE-L, BERTScore" -ForegroundColor Gray
Write-Host "    3c. Ablation study (5 retrieval configs)" -ForegroundColor Gray
Write-Host "    3d. LLM-as-Judge scoring" -ForegroundColor Gray
Write-Host "    3e. Error analysis" -ForegroundColor Gray
Write-Host "    3f. Generate HTML report" -ForegroundColor Gray
Write-Host "  This is the longest step (~30-60 minutes)" -ForegroundColor Yellow

$sw = [System.Diagnostics.Stopwatch]::StartNew()
python ConversationalAI_Assignment_2_Group_122_Part2_Evaluation.py
$sw.Stop()

if ($LASTEXITCODE -ne 0) {
    Write-Err "Evaluation.py failed (exit code $LASTEXITCODE)"
    exit 1
}

# Validate outputs
$outputs = @(
    @{ File = "evaluation_results.csv";  Desc = "Per-question metrics" },
    @{ File = "ablation_results.json";   Desc = "Ablation study" },
    @{ File = "evaluation_report.html";  Desc = "HTML report" }
)

$allPresent = $true
foreach ($o in $outputs) {
    if (Test-Path $o.File) {
        $size = (Get-Item $o.File).Length
        $sizeKB = [math]::Round($size / 1KB, 1)
        Write-Ok "$($o.File) ($($sizeKB) KB) - $($o.Desc)"
    } else {
        Write-Warn "$($o.File) not created - $($o.Desc)"
        $allPresent = $false
    }
}

$elapsed = $sw.Elapsed.ToString("hh\:mm\:ss")
Write-Ok "Evaluation pipeline complete ($elapsed)"

# Quick metrics summary from CSV
if (Test-Path "evaluation_results.csv") {
    try {
        $csv = Import-Csv "evaluation_results.csv"
        Write-Host ""
        Write-Host "  ┌─────────────────────────────────────┐" -ForegroundColor Cyan
        Write-Host "  │       EVALUATION RESULTS SUMMARY    │" -ForegroundColor Cyan
        Write-Host "  ├─────────────────────────────────────┤" -ForegroundColor Cyan

        # Try to extract metric columns if they exist
        $columns = $csv[0].PSObject.Properties.Name
        foreach ($metric in @("mrr", "rouge_l", "bert_score", "MRR", "ROUGE_L", "BERTScore")) {
            if ($metric -in $columns) {
                $values = $csv | ForEach-Object { [double]$_.$metric } | Where-Object { $_ -gt 0 }
                if ($values.Count -gt 0) {
                    $avg = ($values | Measure-Object -Average).Average
                    $avgStr = "{0:F4}" -f $avg
                    Write-Host "  │  $($metric.PadRight(12)) :  $avgStr            │" -ForegroundColor Cyan
                }
            }
        }
        Write-Host "  │  Questions       :  $($csv.Count)                │" -ForegroundColor Cyan
        Write-Host "  └─────────────────────────────────────┘" -ForegroundColor Cyan
    } catch {
        # CSV parsing failed - not critical
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Launch Dashboard
# ═══════════════════════════════════════════════════════════════════════════

if (-not $NoDashboard) {
    Write-Step "4" "Launching Evaluation Dashboard"
    Write-Host "  Starting Streamlit at http://localhost:8501" -ForegroundColor Green
    Write-Host "  Press Ctrl+C to stop" -ForegroundColor Gray
    Write-Host ""
    streamlit run ConversationalAI_Assignment_2_Group_122_Part2_Dashboard.py
} else {
    Write-Host ""
    Write-Host "  Dashboard launch skipped (--NoDashboard)" -ForegroundColor Gray
    Write-Host "  To launch later:" -ForegroundColor Gray
    Write-Host "    streamlit run ConversationalAI_Assignment_2_Group_122_Part2_Dashboard.py" -ForegroundColor White
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  Pipeline Complete!" -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  Output files:" -ForegroundColor Gray
Write-Host "    - evaluation_questions.json  (100 Q&A pairs)" -ForegroundColor White
Write-Host "    - evaluation_results.csv     (per-question metrics)" -ForegroundColor White
Write-Host "    - ablation_results.json      (method comparisons)" -ForegroundColor White
Write-Host "    - evaluation_report.html     (full HTML report)" -ForegroundColor White
Write-Host ""
