# This file is used to combine the originally provided test.json with the spoilerType predictions file from the task1 to be able to use for task2.

# Run this file as -- python3 merger.py path1 path2.
# Path1 = Path to original test.json
# Path2 = Path to output file of Task1.
# Output is stored as /data/input/test-merged.jsonl

import pandas as pd
import sys

path1 = sys.argv[1]
path2 = sys.argv[2]
testFile = pd.read_json(path1, lines=True)
spoilerTypeFile = pd.read_json(path2, lines=True)

outputFile = pd.merge(testFile, spoilerTypeFile, on='id')

# Save the merged dataframe as a new jsonl file
outputFile.to_json('../data/input/test-merged.jsonl', orient='records', lines=True)