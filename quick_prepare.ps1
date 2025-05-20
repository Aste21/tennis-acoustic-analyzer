<#
  Usage
  -----
  .\quick_prepare.ps1  path\to\match.mp4        # extract, slice, then open UI
  .\quick_prepare.ps1  path\to\match.mp4 -NoUI  # just build the clips/CSV
#>

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Video,          # full-length MP4/MKV

    [switch]$NoUI            # if you add -NoUI then Streamlit wont be launched
)

$root   = $PSScriptRoot      # project root (same folder as this script)
$stem   = [IO.Path]::GetFileNameWithoutExtension($Video)

$wav    = Join-Path $root "data\audio\$stem.wav"
$clips  = Join-Path $root "data\clips\$stem"
$csv    = Join-Path $clips "slices.csv"

# make folders if they don’t exist
New-Item -ItemType Directory -Path (Split-Path $wav)  -Force | Out-Null
New-Item -ItemType Directory -Path $clips             -Force | Out-Null

# 1. extract WAV
if (-not (Test-Path $wav)) {
    python audio_labeling_tools.py extract --input "$Video" --output "$wav"
}

# 2. slice audio into wav files and also into mp4 files for easier labeling
python audio_labeling_tools.py slice `
    --input "$wav" `
    --video "$Video" `
    --output_dir "$clips" `
    --csv "$csv" `
    --method onset `
    --delta 0.2 `
    --export_video `
    --min_gap_ms 200 `
    --pre_ms 150 --post_ms 150   # this changes the duration of the audio clip before and after the ,,noise"

# 3. open labelling UI unless -NoUI was supplied
if (-not $NoUI) {
    streamlit run label_app.py
}
