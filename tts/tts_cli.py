#!/usr/bin/env python3
"""
Tiny CLI Text‑to‑Speech with Pyttsx3
  • Interactive *or* one‑shot mode (flags).
  • Minimal overhead: 1 engine init, 1 run loop per utterance.
"""

import argparse
import os
import pyttsx3

# ---------- Constants -------------------------------------------------------

DEFAULT_RATE = 100  # words per minute
SPEED_MENU = {
    "1": ("Very Slow", 60),
    "2": ("Slow", 80),
    "3": ("Normal", 95),
    "4": ("Fast", 100),
    "5": ("Custom", None),  # handled separately
}
MENU_TEXT = "\n".join(
    [
        "\nSpeed Options:",
        *[f"{k}. {name} ({rate or 'custom'} wpm)" for k, (name, rate) in SPEED_MENU.items()],
    ]
)

# ---------- Helpers ---------------------------------------------------------

def choose_speed(engine):
    """Prompt user once, return integer WPM."""
    print(f"\nCurrent speed: {engine.getProperty('rate')} wpm (120‑150 ≈ normal human)")
    while True:
        print(MENU_TEXT)
        choice = input("Select speed (1‑5): ").strip()
        if choice in SPEED_MENU:
            label, rate = SPEED_MENU[choice]
            if rate:
                return rate
            # custom
            try:
                val = int(input("Enter custom speed (60‑200): "))
                if 60 <= val <= 200:
                    return val
                print("Range must be 60‑200")
            except ValueError:
                print("Enter a number")
        else:
            print("Invalid choice")

def speak(engine, text, outfile=None):
    """Queue text and run.  If outfile given, saves WAV instead of speaking."""
    if outfile:
        if not outfile.endswith(".wav"):
            outfile += ".wav"
        engine.save_to_file(text, outfile)
    else:
        engine.say(text)
    engine.runAndWait()
    if outfile:
        print(f"Saved → {os.path.abspath(outfile)}")

def interactive_loop(engine):
    """REPL‑style text‑to‑speech session."""
    print("\n=== Text‑to‑Speech Converter ===")
    print(f"Default speed = {DEFAULT_RATE} wpm (very slow)\nType 'quit' to exit.")
    while True:
        text = input("\nEnter text:\n> ").strip()
        if text.lower() == "quit":
            break
        if not text:
            print("Please enter some text")
            continue

        engine.setProperty("rate", choose_speed(engine))

        print("\nOutput Options:\n1. Speak now\n2. Save to file")
        opt = input("Choose (1‑2): ").strip()
        if opt == "2":
            fname = input("Filename (without extension): ").strip() or "output"
            speak(engine, text, fname)
        else:
            speak(engine, text)

# ---------- Main ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simple Pyttsx3 TTS")
    parser.add_argument("-t", "--text", help="Text to convert (skips interactive prompt)")
    parser.add_argument("-o", "--out", help="Output WAV filename (implies --text)")
    parser.add_argument("-r", "--rate", type=int, help="Speech rate (wpm, 60‑200)")
    args = parser.parse_args()

    engine = pyttsx3.init()
    engine.setProperty("rate", args.rate or DEFAULT_RATE)
    engine.setProperty("volume", 0.9)

    if args.text:
        speak(engine, args.text.strip(), args.out)
    else:
        interactive_loop(engine)

if __name__ == "__main__":
    main()
