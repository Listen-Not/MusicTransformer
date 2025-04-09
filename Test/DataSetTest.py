import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dataSet
import config


test=dataSet.MIDIDataset(config.SAVE_DIR,mode='train')
a=len(test)
test.batch(2,10)
print(test)