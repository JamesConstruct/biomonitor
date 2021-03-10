"""
 * This file is part of the BioMonitor distribution.
 * Copyright (c) 2021 Jakub Telčer.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.

 * See readme.md for usage instructions.
 """

import argparse

import camera_input as camera
from classifier import Classifier
from config import MainConfig, CameraConfig, IsolatorConfig
from object_isolator import ObjectIsolator
from rating import Rating


parser = argparse.ArgumentParser(prog="BioMonitor",
                                 description="Real-time bio-monitoring of freshwater contamination by analysis of "
                                             "movement of Daphnia magna.")
parser.add_argument("-c", "--cam", dest="cam_index", help="Select camera (input the device index)",
                    type=int, default=0)
parser.add_argument("-f", "--file", dest="source_file", help="The source file will be used instead of camera input")
parser.add_argument("-fps", "--fps", dest="fps", help="Set the framerate (default 30)", type=int, default=30)
parser.add_argument("-d", "--debug", dest="debug", help="Show video and drawn bounding boxes", action="store_true")
parser.add_argument("-v", "--verbose", dest="verbose", help="Increase verbosity of the program, print object counts.",
                    action="store_true")
parser.add_argument("-p", "--predictions", dest="predictions", help="Print predictions", action="store_true")
parser.add_argument("-dw", "--dontwait", dest="dontwait", help="Prevent program from waiting for frames",
                    action="store_true")


def main():
    args = parser.parse_args()
    print("\nWelcome to BioMonitor!\n\nUse Ctrl+C to quit\nStarting...")

    CameraConfig.simulation_source = args.source_file
    CameraConfig.index = args.cam_index
    IsolatorConfig.debug = args.debug
    if args.dontwait:
        args.fps = 99999999

    cam = camera.CameraInput(args.fps, MainConfig.buffer_size, bool(args.source_file))
    ai = Classifier()
    ai.load()

    buffer = cam.get()

    while buffer:
        trajectories = ObjectIsolator.isolate(buffer)
        prepared = ai.prepare(trajectories)

        predictions = []
        for one in prepared:
            predictions.append(ai.predict(one, args.predictions))

        if args.verbose:
            print(f"\nDetected: \t{len(predictions)}\nContaminated: \t{sum([x.argmax() for x in predictions])}")

        print("Rating:\t\t", Rating.evaluate(predictions))

        buffer = cam.get()

    cam.terminate()


if __name__ == "__main__":
    main()
