$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:UV_CACHE_DIR=Join-Path (Get-Location) '.tools/uv-cache'
$env:UV_PYTHON_INSTALL_DIR=Join-Path (Get-Location) '.tools/python'
$uv=Get-Command uv -ErrorAction SilentlyContinue
if ($uv) { $uvExe=$uv.Source }
elseif (Test-Path '.tools/uv/uv.exe') { $uvExe=(Resolve-Path '.tools/uv/uv.exe').Path }
else { throw 'uv is unavailable. Install uv before rerunning this setup script.' }
& $uvExe venv .venv --python 3.12
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
& $uvExe pip compile requirements.in --python .venv/Scripts/python.exe --output-file requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'dependency resolution failed' }
& $uvExe pip sync requirements.txt --python .venv/Scripts/python.exe
if ($LASTEXITCODE -ne 0) { throw 'dependency installation failed' }
& .venv/Scripts/python.exe -c 'import pandas,numpy,geopandas,shapely,pyproj,pyogrio,lifelines,matplotlib,dotenv,openpyxl; print(1)'
if ($LASTEXITCODE -ne 0) { throw 'import verification failed' }

