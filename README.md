# tennis-events-acoustic-data

To run, enter into cmd: .\quick_prepare.ps1 .\input_videos\name_of_clip.mp4

streamlit run label_app.py


3,5 h video -> splicing takes ~20 minutes 

for later:

1. Add ball bounce as a label
2. Maybe add something like racket hitting the ground as a label?
3. Also later on we gotta think about ball hitting the net, its rare but potentially important
4. What to do with foot hitting the ground? sound like ball being hit but is something much different.
5. Maybe yell/grunt should be also added.
6. In POC, interviews are also as a commentator coze of similarity
## Audio inference

After training the acoustic classifier, place the weights at `models/acoustic_classifier.pth`. You can detect events in a WAV file using:

```bash
python audio_inference.py --audio path/to/file.wav --output detections.json
```

The script outputs a JSON file with timestamps and predicted labels for each detected sound event.
