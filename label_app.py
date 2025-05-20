import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd
import soundfile as sf
import streamlit as st
import label_config as cfg

# ───────────────────────────── helpers ────────────────────────────────────
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def backup_csv(src: Path, session: str):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = BACKUP_DIR / f"{session}_{ts}.csv"
    shutil.copy(src, dst)

# ───────────────────────────── choose session ─────────────────────────────
CLIPS_ROOT = Path(cfg.CLIPS_ROOT)
video_dirs = sorted(p for p in CLIPS_ROOT.iterdir() if (p / "slices.csv").exists())
if not video_dirs:
    st.error(f"No folders with slices.csv inside {CLIPS_ROOT}")
    st.stop()

with st.sidebar:
    session_dir = st.selectbox("🎞  Choose session", video_dirs, format_func=lambda p: p.name)
    auto_loop   = st.checkbox("Loop / auto-play video", value=True)

CSV_PATH   = session_dir / "slices.csv"
BACKUP_DIR = Path(cfg.BACKUP_DIR); BACKUP_DIR.mkdir(exist_ok=True)

# ───────────────────────────── load table ────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["label"] = df["label"].fillna("").astype(str)

unlab      = df[df["label"] == ""]
total, done = len(df), len(df) - len(unlab)

st.title("🎾 Tennis-Sound Labeller")
st.caption(f"Session **{session_dir.name}** — {done}/{total} clips ({done/total:.1%})")
st.progress(done / total)

if unlab.empty:
    st.success("🎉  All clips in this session are done!")
    st.stop()

row       = unlab.iloc[0]
clip_id   = row["clip_id"]
wav_path  = session_dir / clip_id
mp4_path  = (session_dir / row["video_id"]) if row["video_id"] else wav_path.with_suffix(".mp4")

# remember which clip is on screen
st.session_state["current_clip_id"] = clip_id
# ───────────────────────────── playback ──────────────────────────────────
st.markdown(f"Now labelling: **{clip_id}**")

if mp4_path.exists():
    st.video(str(mp4_path))          # normal Streamlit player

    # Global hot-key: press **P** anywhere to play / pause the clip -> =makes it much easier o label everything
    st.components.v1.html(
        """
        <script>
          (() => {
            // Do nothing while typing in inputs
            function typingIn(el) {
              const t = (el.tagName || '').toLowerCase();
              return t === 'input' || t === 'textarea' || el.isContentEditable;
            }

            window.parent.document.addEventListener(
              'keydown',
              (e) => {
                if (e.key.toLowerCase() !== 'p' || typingIn(e.target)) return;
                e.preventDefault();      // stop Streamlit widgets seeing the key
                e.stopPropagation();

                const vid = window.parent.document.querySelector('video');
                if (vid) {
                  (vid.paused ? vid.play() : vid.pause()).catch(() => {});
                }
              },
              true      // capture phase → runs before any widget handler
            );
          })();
        </script>
        """,
        height=0,
    )

    st.caption("Press **P** anywhere to play / pause (works even after clicking a label).")
else:
    audio, sr = sf.read(wav_path, dtype="float32")
    st.audio(audio, format="audio/wav", sample_rate=sr)


# ───────────────────────────── label controls ────────────────────────────
cols   = st.columns(len(cfg.LABELS))
choice = None
for c, lab in zip(cols, cfg.LABELS):
    if c.button(lab):
        choice = lab

# ───────────────────────────── save / advance ────────────────────────────
if choice:
    backup_csv(CSV_PATH, session_dir.name)
    sel_id = st.session_state["current_clip_id"]
    df.loc[df["clip_id"] == sel_id, "label"] = choice
    df.to_csv(CSV_PATH, index=False)
    safe_rerun()

# ───────────────────────────── undo panel ────────────────────────────────
with st.expander("Undo last label"):
    done_rows = df[df["label"] != ""]
    if done_rows.empty:
        st.write("Nothing to undo yet.")
    else:
        last = done_rows.tail(1).iloc[0]
        st.write(last["clip_id"], "→", last["label"])
        if st.button("Clear this label"):
            backup_csv(CSV_PATH, session_dir.name)
            df.loc[df["clip_id"] == last["clip_id"], "label"] = ""
            df.to_csv(CSV_PATH, index=False)
            safe_rerun()
