<#
==============================================================================
Network-Aware Compression - Windows Receiver Setup Script
Target OS: Windows 10 / Windows 11
==============================================================================
#>

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " Network-Aware Compression Receiver - Windows Setup & Validation" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# 1. Locate Python
$PythonExe = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonExe) {
    $PythonExe = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $PythonExe) {
    Write-Host "[ERROR] Python 3.10+ was not found on your system PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10 or higher from https://www.python.org/downloads/ and check 'Add Python to PATH'." -ForegroundColor Yellow
    Exit 1
}

$pyVer = & $PythonExe.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Host "[INFO] Detected Python version: $pyVer ($($PythonExe.Source))" -ForegroundColor Green

# 2. Virtual Environment Setup
$VenvDir = Join-Path $PSScriptRoot "venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] Creating virtual environment at $VenvDir..." -ForegroundColor Yellow
    & $PythonExe.Source -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        Exit 1
    }
} else {
    Write-Host "[INFO] Virtual environment already exists at $VenvDir." -ForegroundColor Green
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# 3. Upgrade pip and install requirements
Write-Host "[INFO] Upgrading pip..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip --quiet

if (Test-Path (Join-Path $PSScriptRoot "requirements.txt")) {
    Write-Host "[INFO] Installing / verifying dependencies from requirements.txt..." -ForegroundColor Yellow
    & $VenvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt") --quiet
} else {
    Write-Host "[WARN] requirements.txt not found." -ForegroundColor Yellow
}

# 4. Check PyTorch & Hardware Acceleration
Write-Host "[INFO] Verifying PyTorch and CUDA runtime..." -ForegroundColor Yellow
$TorchCheckScript = @"
import torch
print(f'PyTorch Version: {torch.__version__}')
cuda_avail = torch.cuda.is_available()
print(f'CUDA Available: {cuda_avail}')
if cuda_avail:
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'Device Count: {torch.cuda.device_count()}')
    t = torch.randn(2, 2, device='cuda')
    print('CUDA Tensor Allocation: SUCCESS')
else:
    print('Hardware Acceleration: Running in CPU-only mode.')
"@
& $VenvPython -c $TorchCheckScript

# 5. Check Model Checkpoint
Write-Host "`n[INFO] Validating U-Net Model Checkpoints..." -ForegroundColor Yellow
$ModelPth = Join-Path $PSScriptRoot "models\best_unet.pth"
$ModelPt = Join-Path $PSScriptRoot "models\best_unet.pt"

if (Test-Path $ModelPth) {
    $sizeMb = [math]::Round((Get-Item $ModelPth).Length / 1MB, 2)
    Write-Host "  [OK] PyTorch weights found: $ModelPth ($sizeMb MB)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] models/best_unet.pth not found." -ForegroundColor Yellow
}

if (Test-Path $ModelPt) {
    $sizeMb = [math]::Round((Get-Item $ModelPt).Length / 1MB, 2)
    Write-Host "  [OK] TorchScript checkpoint found: $ModelPt ($sizeMb MB)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] models/best_unet.pt not found. Run model conversion or verify training." -ForegroundColor Yellow
}

# 6. Windows Firewall Guidance
Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host " Windows Firewall Notice for Inbound TCP Traffic" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "To allow the Ubuntu Sender to connect to this Windows Receiver on port 5000:" -ForegroundColor Yellow
Write-Host "Run the following command once in an ELEVATED (Administrator) PowerShell prompt:" -ForegroundColor White
Write-Host 'New-NetFirewallRule -DisplayName "NetworkAwareCompression-Receiver" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow' -ForegroundColor Cyan
Write-Host "`nTo find your Windows local IP address to enter into sender's config.json, run:" -ForegroundColor White
Write-Host "ipconfig | findstr /i IPv4" -ForegroundColor Cyan

Write-Host "`n[SUCCESS] Windows Receiver setup and verification completed!" -ForegroundColor Green
Write-Host "You can start the receiver application anytime using: .\run_receiver.bat" -ForegroundColor Green
