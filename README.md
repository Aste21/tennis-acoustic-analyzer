# Tennis Acoustic Analyzer

This project provides tools for preparing audio clips and labeling tennis sounds. A new `analyze_match.py` script combines an acoustic hit detector with video models to process full matches.

## Quick start

```bash
# prepare clips from a video
./quick_prepare.ps1 ./input_videos/clip.mp4

# run labeling app
streamlit run label_app.py

# analyze a full match
python analyze_match.py path/to/video.mp4 --models_dir models --output_csv results.csv
```

The `models` directory should contain pretrained weights:

- `acoustic_classifier.pth`
- `ball_track_model.pth`
- `court_model.pth`
- `bounce_model.cbm`
