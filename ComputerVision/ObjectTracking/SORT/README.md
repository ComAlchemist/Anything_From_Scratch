# Sort-Object-Tracking

Source: <https://github.com/arunm8489/Sort-Object-Tracking/tree/main>

Implementation of Simple Online and Realtime Tracking using Python. It is designed for multiple object tracking where only current and previous frames are available .While this minimalistic tracker doesn't handle occlusion or re-entering objects its purpose is to serve as a baseline and testbed for the development of future trackers. <br>
paper : <https://arxiv.org/abs/1602.00763>

## Usage

```
pip install -r requirements.txt
```

```
from sort.sort import *
tracker = Sort()
tracker.update()
```

### Sample output with Yolov3

<img src="https://github.com/arunm8489/Sort-Object-Tracking/blob/main/images/track1.png">
<br>
<br>
<img src = "https://github.com/arunm8489/Sort-Object-Tracking/blob/main/images/track2.png">
