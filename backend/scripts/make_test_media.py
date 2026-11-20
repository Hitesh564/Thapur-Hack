"""Generate original test media (images via OpenAI, then slideshow MP4s) for pipeline tests.

Outputs in tests/fixtures/media/:
  cooking_weak.mp4   - static, dull opening, then the cooking shots (baseline)
  cooking_strong.mp4 - same body, opens on the cheese pull (alternate opening)
  gaming.mp4         - gaming desk shots, no audio
Run:  python scripts/make_test_media.py   (needs OPENAI_API_KEY; images are cached)
"""
import base64
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ingestion.upload import ffmpeg_exe  # noqa: E402
from app.video_analysis.vlm import get_client  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "media"
STYLE = "Vertical smartphone video still, photorealistic, natural light, no text, no logos."
IMAGES = {
    "c_weak": "A plain empty white bowl on a cluttered kitchen counter, flat dull lighting, nothing happening.",
    "c_garlic": "Close-up of minced garlic sizzling in melted butter in a frying pan, home kitchen.",
    "c_toss": "Overhead shot of spaghetti being tossed in a pan of garlic butter sauce, steam rising.",
    "c_pull": "Extreme close-up of a fork twirling glossy garlic butter spaghetti with a long stretchy parmesan cheese pull.",
    "c_plate": "Finished plate of garlic butter spaghetti with parsley on a wooden table, appetising.",
    "g_desk": "A young adult at a desk with RGB lighting playing a competitive first-person shooter on a large monitor, face lit by the screen.",
    "g_keys": "Close-up of hands on a glowing RGB mechanical gaming keyboard and mouse during intense play.",
    "g_win": "A gaming monitor showing a generic colourful 'victory' screen with confetti, desk in the foreground.",
}
NARRATION = ("This garlic butter pasta takes fifteen minutes and one pan. Sizzle the garlic in butter, "
             "toss in the spaghetti, add parmesan, and dinner is done.")
W, H, FPS = 720, 1280, 30


def gen_image(name: str, prompt: str) -> Path:
    p = OUT / f"{name}.jpg"
    if p.exists():
        return p
    r = get_client().images.generate(model="gpt-image-2", prompt=f"{prompt} {STYLE}", size="1024x1536")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(r.data[0].b64_json), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(p), cv2.resize(img, (W, H)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p


def shot(img: np.ndarray, seconds: float, zoom: float, still: bool = False):
    n = int(seconds * FPS)
    for i in range(n):
        s = 1.0 if still else 1.0 + zoom * i / n
        m = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        yield cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REFLECT)


def make_video(name: str, plan: list[tuple[str, float, float, bool]], audio: Path | None) -> Path:
    silent = OUT / f"{name}_silent.mp4"
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for img_name, secs, zoom, still in plan:
        for f in shot(cv2.imread(str(OUT / f"{img_name}.jpg")), secs, zoom, still):
            vw.write(f)
    vw.release()
    out = OUT / f"{name}.mp4"
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent)]
    if audio:
        cmd += ["-i", str(audio), "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(out)]
    subprocess.run(cmd, check=True)
    silent.unlink()
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda kv: gen_image(*kv), IMAGES.items()))
    voice = OUT / "narration.mp3"
    if not voice.exists():
        r = get_client().audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=NARRATION)
        voice.write_bytes(r.content)
    body = [("c_garlic", 3.5, 0.10, False), ("c_toss", 3.5, 0.12, False), ("c_plate", 3.0, 0.06, False)]
    make_video("cooking_weak", [("c_weak", 3.5, 0, True)] + body, voice)
    make_video("cooking_strong", [("c_pull", 1.5, 0.25, False), ("c_plate", 1.0, 0.1, False)] + body, voice)
    make_video("gaming", [("g_desk", 3.0, 0.1, False), ("g_keys", 3.0, 0.15, False), ("g_win", 3.0, 0.08, False)], None)
    print("done:", sorted(p.name for p in OUT.glob("*.mp4")))


if __name__ == "__main__":
    main()
