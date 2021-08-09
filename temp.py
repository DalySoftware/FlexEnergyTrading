import numpy as np
import talib  # type: ignore
from talib import *
from talib import MA_Type

# print(talib.get_functions())


close = np.random.random(100)

print(close)

output = talib.SMA(close)
# print(output)

print(talib.BBANDS(close, matype=MA_Type.T3))
