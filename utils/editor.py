character_to_add = "\"}"  # Replace with your specific character

with open('/media/nikunj/DATA1/WORK/Uwaterloo/Spring_2023/MSCI_641/Project/final/task1/nancy-hicks-gribble-at-SemEval-2023-Task-5/Data/test-output.jsonl', 'r') as input_file, open('output.txt', 'w') as output_file:
    for line in input_file:
        # rstrip() removes trailing whitespace and newline characters
        # We add our specific character, and then add the newline character back
        modified_line = line.rstrip() + character_to_add + '\n'
        output_file.write(modified_line)
