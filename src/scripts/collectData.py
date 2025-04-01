import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from utils.datacollectors import DataCollector




if __name__ == "__main__":
    collectionConfigs = [
        # configs, frequency, output file
        ("../../config/trainingConfigs/circle.json", 25, "cylinderData", 2000)
    ]

    dc = DataCollector(collectionConfigs)
    dc.collect()
