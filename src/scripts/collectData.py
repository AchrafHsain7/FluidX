import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from utils.datacollectors import DataCollector




if __name__ == "__main__":
    collectionConfigs = [
        ("../../config/trainingConfigs/airplane2D.json", 25, "data1")
    ]

    dc = DataCollector(collectionConfigs)
    dc.collect()
