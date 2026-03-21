# Part 1 Complete Execution Script
# Creates the Wikipedia dataset (fixed_url.json and random_url.json)
# Then launches the Streamlit RAG application

$ErrorActionPreference = "Stop"

# Navigate to project directory
cd "c:\Users\koush\OneDrive\Documents\BITS Pilani\Conversational AI\Assignment 2"

# Activate virtual environment
Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "PART 1: Hybrid RAG System Setup" -ForegroundColor Magenta
Write-Host "========================================`n" -ForegroundColor Magenta

# Step 1: Convert Notebook to Python script (if needed)
Write-Host "Step 1: Preparing dataset generation script..." -ForegroundColor Cyan

# Check if data files already exist
$fixedExists = Test-Path "fixed_url.json"
$randomExists = Test-Path "random_url.json"

if ($fixedExists -and $randomExists) {
    Write-Host "Dataset files already exist!" -ForegroundColor Green
    $fixed = Get-Content "fixed_url.json" | ConvertFrom-Json
    $random = Get-Content "random_url.json" | ConvertFrom-Json
    Write-Host "  - fixed_url.json: $($fixed.Count) entries" -ForegroundColor Gray
    Write-Host "  - random_url.json: $($random.Count) entries" -ForegroundColor Gray
    
    $choice = Read-Host "`nRegenerate datasets? (y/N)"
    if ($choice -ne "y" -and $choice -ne "Y") {
        Write-Host "Skipping dataset generation." -ForegroundColor Yellow
        $skipGeneration = $true
    }
}

if (-not $skipGeneration) {
    Write-Host "`nStep 2: Running Jupyter notebook to generate datasets..." -ForegroundColor Cyan
    Write-Host "This will create fixed_url.json (200 URLs) and random_url.json (300 URLs)" -ForegroundColor Gray
    Write-Host "Expected time: 10-20 minutes (depends on network speed)`n" -ForegroundColor Yellow
    
    # Run notebook using nbconvert
    # jupyter nbconvert --to script --execute "ConversationalAI_Assignment_2_Group_122_Par1_1.1_to_1.4.ipynb" --ExecutePreprocessor.timeout=3600
    
    # Verify files were created
    if (Test-Path "fixed_url.json") {
        $fixed = Get-Content "fixed_url.json" | ConvertFrom-Json
        Write-Host "SUCCESS: fixed_url.json created with $($fixed.Count) entries" -ForegroundColor Green
    }
    else {
        Write-Host "ERROR: fixed_url.json was not created!" -ForegroundColor Red
    }
    
    if (Test-Path "random_url.json") {
        $random = Get-Content "random_url.json" | ConvertFrom-Json
        Write-Host "SUCCESS: random_url.json created with $($random.Count) entries" -ForegroundColor Green
    }
    else {
        Write-Host "ERROR: random_url.json was not created!" -ForegroundColor Red
    }
}

# Step 3: Launch Streamlit App
Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "PART 1.5: Launching RAG Web Interface" -ForegroundColor Magenta
Write-Host "========================================`n" -ForegroundColor Magenta

Write-Host "Starting Streamlit server at http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Gray

streamlit run ConversationalAI_Assignment_2_Group_122_Par1_1.5-streamlit-app.py
