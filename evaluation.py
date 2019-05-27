#!/usr/bin/env python
import sys
import os
import os.path

[_, submission_path, truth_path] = sys.argv

# read submission file
if os.path.exists(submission_path) and submission_path.endswith('submission.txt'):
    with open(submission_path) as submission_file:
        submission = submission_file.readlines()
else:
    sys.exit("submission.txt file not found")

# read truth file
if os.path.exists(truth_path):
    if truth_path.endswith('train_norm.txt'):
        with open(truth_path) as truth_file:
            truth = truth_file.readlines()
    elif truth_path.endswith('test_norm.txt'):
        with open(truth_path) as truth_file:
            truth = truth_file.readlines()
    else:
        sys.exit("train_norm.txt or test_norm.txt file not found")
else:
    sys.exit("train_norm.txt or test_norm.txt file not found")

# check number of mentions between gold and submission is equal
if len(truth) != len(submission):
    message = "Expected number of mentions {0}, found {1}"
    sys.exit(message.format(len(truth), len(submission)))

# output accuracy
count_correct = 0
for idx, _ in enumerate(truth):
    if truth[idx].strip() == submission[idx].strip():
        count_correct += 1
score = count_correct / float(len(truth))
print("accuracy:{0}\n".format(score))
