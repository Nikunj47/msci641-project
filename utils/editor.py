character_to_add = "\"}"  # Replace with your specific character

with open('../data/input/{filename}.jsonl', 'r') as input_file, open('output.txt', 'w') as output_file:
    for line in input_file:
        # rstrip() removes trailing whitespace and newline characters
        # We add our specific character, and then add the newline character back
        modified_line = line.rstrip() + character_to_add + '\n'
        output_file.write(modified_line)
