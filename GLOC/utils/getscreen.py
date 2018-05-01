from sys import platform

# MacOS X
if platform[:3] == 'dar':
    from mss.darwin import MSS as mss
# GNU/Linux
elif platform[:3] == 'lin':
    from mss.linux import MSS as mss
# Microsoft Windows
elif platform[:3] == 'win':
    from mss.windows import MSS as mss

import mss.tools

def getScreen(bbox = (0,0,800,640)):
    with mss.mss() as sct:
        # Use the 1st monitor
        monitor = sct.monitors[1]
        return sct.grab(bbox)

        # mss.tools.to_png(im.rgb, im.size, 'blah.png')
