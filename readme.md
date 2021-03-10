# Welcome to BioMonitor #
This piece of software detects freshwater contamination in near real-time using optical analysis of movement of Daphnia magna.

## How do I run it? ##
To run this software [Python 3.8+](https://www.python.org/downloads/) is required.

Install dependencies:
```
pip install -r requirements.txt
```

And then run the program:
```
python Biomonitor
```

You can also install this package and use its components:
```
pip install -e Biomonitor
```

There are few parameters that can be set in the ```config.py``` or via the CLI:

Parameter | Value | Description
--------- | ----- | -----------
-h, --help | | Show help message
-c, --cam | int | Set camera device index
-f, --file | string | Specify source file instead of camera input
-fps, --fps | int | Set framerate
-d, --debug | | Enter debug mode, show the video with drawn bounding boxes
-v, --verbose | | Increase the verbosity of the program, print object counts
-p, --predictions | | Show detailed list of made predictions
-dw, --dontwait | | Prevent program from waiting for frames

All the parameters are optional.

## License ##
Copyright (c) 2021 Jakub Telčer.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

See LICENSE for further information.