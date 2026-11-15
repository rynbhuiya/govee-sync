#!/usr/bin/env python3
"""Sync a Govee lamp's color to the album art of the currently playing Spotify track.

Usage:
    python spotify_govee_sync.py                  # single dominant color (default)
    python spotify_govee_sync.py --mode gradient  # cover palette spread across the lamp's segments
    python spotify_govee_sync.py --list-devices   # print your Govee devices (sku + device id)
    python spotify_govee_sync.py --once           # set the color once and exit
"""

import argparse
import colorsys
import io
import os
import sys
import time
import uuid

import requests
import spotipy
from dotenv import load_dotenv
from PIL import Image
from spotipy.oauth2 import SpotifyOAuth

GOVEE_BASE = "https://openapi.api.govee.com/router/api/v1"
POLL_SECONDS = 3
GREY_SATURATION = 0.12  # below this a color counts as grey


def require_env(name):
    value = os.getenv(name)
    if not value:
        sys.exit(f"Missing {name}. Set it in .env (see .env.example).")
    return value


# ---------- Spotify ----------

def spotify_client():
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=require_env("SPOTIFY_CLIENT_ID"),
            client_secret=require_env("SPOTIFY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
            scope="user-read-currently-playing",
            cache_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spotify_cache"),
        )
    )


def current_track(sp):
    """Return (key, label, image_url) for the playing track, or None if nothing is playing."""
    playing = sp.currently_playing(additional_types="episode")
    if not playing or not playing.get("is_playing") or not playing.get("item"):
        return None
    item = playing["item"]
    # Tracks keep art on the album; podcast episodes keep it on the item itself.
    container = item.get("album") or item
    images = container.get("images") or []
    if not images:
        return None
    # Images are sorted largest first; the smallest is plenty for color extraction.
    url = images[-1]["url"]
    artists = ", ".join(a["name"] for a in item.get("artists", []))
    label = f"{item['name']} - {artists}" if artists else item["name"]
    # Key on the art, not the track, so songs from the same album don't re-send.
    return url, label, url


# ---------- Color ----------

def extract_palette(image_bytes):
    """Quantize the cover to 8 colors.

    Returns [(score, mean_y, (h, s, v)), ...] sorted best first, where mean_y is
    the color's average vertical position in the cover (0 = top, 1 = bottom).
    """
    size = 64
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    palette_img = img.quantize(colors=8)
    palette = palette_img.getpalette()

    stats = {}  # palette index -> [pixel_count, sum_of_rows]
    for i, idx in enumerate(palette_img.tobytes()):
        entry = stats.setdefault(idx, [0, 0])
        entry[0] += 1
        entry[1] += i // size

    entries = []
    for idx, (count, sum_y) in stats.items():
        r, g, b = palette[idx * 3: idx * 3 + 3]
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        # Weight by area, but favor saturated, reasonably bright colors so a
        # black/white/grey background doesn't beat the color that defines the cover.
        score = count * (0.1 + s) * (0.2 + v)
        entries.append((score, sum_y / count / (size - 1), (h, s, v)))
    entries.sort(reverse=True)
    return entries


def lamp_rgb(hsv):
    """Lamps render dark/muddy colors poorly: push to full brightness and lift saturation."""
    h, s, v = hsv
    r, g, b = colorsys.hsv_to_rgb(h, min(1.0, s * 1.3), 1.0)
    return (round(r * 255), round(g * 255), round(b * 255))


def dominant_color(image_bytes):
    """Pick the most prominent *colorful* color in the image as an (r, g, b) tuple."""
    hsv = extract_palette(image_bytes)[0][2]
    if hsv[1] < GREY_SATURATION:
        # Genuinely greyscale cover: use a neutral white rather than a random tint.
        return (255, 255, 255)
    return lamp_rgb(hsv)


def gradient_colors(image_bytes, n):
    """Pick up to n colorful palette colors, ordered as they appear top-to-bottom in the cover."""
    entries = extract_palette(image_bytes)
    total = sum(score for score, _, _ in entries)
    colorful = [
        e for e in entries
        if e[2][1] >= GREY_SATURATION and e[2][2] >= 0.15 and e[0] >= total * 0.02
    ]
    if not colorful:
        return [(255, 255, 255)]
    # Shades of one hue look identical once brightened, so keep only the best of each.
    distinct = []
    for e in colorful:
        hue = e[2][0]
        if all(min(abs(hue - d[2][0]), 1 - abs(hue - d[2][0])) > 0.04 for d in distinct):
            distinct.append(e)
    picked = sorted(distinct[:n], key=lambda e: e[1])
    return [lamp_rgb(hsv) for _, _, hsv in picked]


# ---------- Govee ----------

def govee_headers():
    return {"Govee-API-Key": require_env("GOVEE_API_KEY"), "Content-Type": "application/json"}


def fetch_devices():
    resp = requests.get(f"{GOVEE_BASE}/user/devices", headers=govee_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])


def list_devices():
    devices = fetch_devices()
    if not devices:
        print("No devices found on this Govee account.")
    for d in devices:
        print(f"{d.get('deviceName', '?'):30}  GOVEE_SKU={d['sku']}  GOVEE_DEVICE={d['device']}")


def segment_info():
    """Return (segment_count, max_segments_per_command) for the configured lamp."""
    override = os.getenv("GOVEE_SEGMENTS")
    device_id = require_env("GOVEE_DEVICE")
    for d in fetch_devices():
        if d["device"] != device_id:
            continue
        for cap in d.get("capabilities", []):
            if cap.get("instance") != "segmentedColorRgb":
                continue
            for field in cap.get("parameters", {}).get("fields", []):
                if field.get("fieldName") == "segment":
                    rng = field.get("elementRange") or {}
                    if "max" in rng:
                        count = rng["max"] - rng.get("min", 0) + 1
                        # The max array size is both the per-command cap and, on lamps
                        # like the H6076, the real segment count: it advertises indexes
                        # 0-14 but only 0-6 light anything up.
                        batch = (field.get("size") or {}).get("max", count)
                        count = min(count, batch)
                        return (int(override) if override else count), batch
        if override:
            return int(override), int(override)
        sys.exit("This Govee device doesn't report segment color support; use --mode single "
                 "(or set GOVEE_SEGMENTS in .env if you know its segment count).")
    sys.exit(f"GOVEE_DEVICE {device_id} was not found on this Govee account.")


def govee_control(capability):
    body = {
        "requestId": str(uuid.uuid4()),
        "payload": {
            "sku": require_env("GOVEE_SKU"),
            "device": require_env("GOVEE_DEVICE"),
            "capability": capability,
        },
    }
    resp = requests.post(f"{GOVEE_BASE}/device/control", headers=govee_headers(), json=body, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 200:
        raise RuntimeError(f"Govee rejected the command: {data}")


def rgb_int(rgb):
    r, g, b = rgb
    return (r << 16) | (g << 8) | b


def set_lamp_color(rgb):
    govee_control({
        "type": "devices.capabilities.color_setting",
        "instance": "colorRgb",
        "value": rgb_int(rgb),
    })


def set_lamp_gradient(colors, segments, batch):
    """Spread colors across the lamp as contiguous bands, first color on segment 0.

    batch is the most segments the lamp accepts in a single command.
    """
    groups = {}  # rgb -> [segment indexes]
    for seg in range(segments):
        groups.setdefault(colors[seg * len(colors) // segments], []).append(seg)
    for rgb, segs in groups.items():
        for i in range(0, len(segs), batch):
            govee_control({
                "type": "devices.capabilities.segment_color_setting",
                "instance": "segmentedColorRgb",
                "value": {"segment": segs[i:i + batch], "rgb": rgb_int(rgb)},
            })


def hex_color(rgb):
    return "#%02x%02x%02x" % rgb


# ---------- Main ----------

def sync_once(sp, last_key, show):
    """Update the lamp if the album art changed. Returns the current art key."""
    track = current_track(sp)
    if track is None:
        return last_key
    key, label, url = track
    if key == last_key:
        return last_key
    art = requests.get(url, timeout=10)
    art.raise_for_status()
    print(f"{label}  ->  {show(art.content)}")
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["single", "gradient"], default="single",
                        help="single: one dominant color (default); "
                             "gradient: cover palette across the lamp's segments")
    parser.add_argument("--colors", type=int, default=5,
                        help="max palette colors in gradient mode (default 5)")
    parser.add_argument("--reverse", action="store_true",
                        help="gradient mode: put the top of the cover at the bottom of the lamp")
    parser.add_argument("--list-devices", action="store_true", help="list Govee devices and exit")
    parser.add_argument("--once", action="store_true", help="sync once and exit")
    args = parser.parse_args()
    if args.colors < 1:
        parser.error("--colors must be at least 1")

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    if args.list_devices:
        list_devices()
        return

    if args.mode == "gradient":
        segments, batch = segment_info()

        def show(image_bytes):
            colors = gradient_colors(image_bytes, min(args.colors, segments))
            # Segment 0 is the bottom of the lamp, so by default flip the colors
            # to put the top of the cover at the top of the lamp.
            if not args.reverse:
                colors.reverse()
            set_lamp_gradient(colors, segments, batch)
            return " ".join(hex_color(c) for c in reversed(colors)) + "  (lamp top -> bottom)"
    else:
        def show(image_bytes):
            rgb = dominant_color(image_bytes)
            set_lamp_color(rgb)
            return hex_color(rgb)

    sp = spotify_client()

    if args.once:
        if sync_once(sp, None, show) is None:
            print("Nothing is playing right now.")
        return

    print(f"Syncing lamp to Spotify album art ({args.mode} mode). Ctrl+C to stop.")
    last_key = None
    while True:
        try:
            last_key = sync_once(sp, last_key, show)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # keep running through network blips / rate limits
            print(f"[warn] {exc}", file=sys.stderr)
            time.sleep(10)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
