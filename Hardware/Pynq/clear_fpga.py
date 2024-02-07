import os

if os.environ['BOARD'] == 'Ultra96':
	from pynq.overlays.sensors96b import Sensors96bOverlay
	overlay = Sensors96bOverlay('sensors96b.bit')
elif os.environ['BOARD'] == 'ZCU104':
	import pynq
	ol = pynq.Overlay("base.bit")
else:
    raise RuntimeError("Board not supported")