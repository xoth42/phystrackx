"""
phystrackx.py

Entry point for launching the PhysTrackX GUI application.

Author: Isam Balghari
"""

import argparse
import os

import customtkinter as ctk
from gui.menu import MenuScreen
from core import setup_logging


class GUI:
    """Main GUI launcher for PhysTrackX."""

    def __init__(self, skip_intro: bool = False, video_path: str | None = None) -> None:
        self.skip_intro = skip_intro
        self.video_path = video_path
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

    def start(self) -> None:
        """Initializes and starts the main GUI loop."""
        self.menu = MenuScreen(skip_intro=self.skip_intro, video_path=self.video_path)
        self.menu.root.mainloop()


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Launch the PhysTrackX GUI.")
    parser.add_argument("--skip-intro", action="store_true",
                        help="Skip the startup animation and music.")
    parser.add_argument("--video", metavar="PATH",
                        help="Load this video on startup and skip the start button.")
    args = parser.parse_args()
    if args.video:
        args.video = os.path.abspath(args.video)
        if not os.path.isfile(args.video):
            parser.error(f"video file not found: {args.video}")
    return args


def main() -> None:
    """Runs the PhysTrackX GUI application."""
    args = parse_args()
    setup_logging()
    program = GUI(skip_intro=args.skip_intro, video_path=args.video)
    program.start()


if __name__ == "__main__":
    main()
