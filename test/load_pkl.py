import pickle
import sys
input_name = sys.argv[1]
file_name = "cache/" + input_name
with open(file_name,"rb") as f:
    train_example = pickle.load(f)
    print(train_example)
    print(len(train_example))
