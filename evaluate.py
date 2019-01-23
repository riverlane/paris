#! /usr/bin/env python3

# Qiskit and sklearn have warnings.
import warnings
warnings.filterwarnings("ignore")

import argparse
import pickle
import sys
import os
import time
import datetime
import json
import re
import inspect
import traceback
import numpy as np

import example_solutions as trialmodule
from example_solutions.helper_functions import print_circuit

parser = argparse.ArgumentParser(description='Tests your solutions for the quantum classification problem.')
parser.add_argument('--solution_function_name', "--fun", metavar='S', type=str,
                    help='The name of your function in small_circuits.py')
parser.add_argument('--print_problem_stats', "--stats", action='store_true',
                    help='Prints some statistics about the training data and the problem.')
parser.add_argument('--cheat', action='store_true',
                    help='Prints the transformation circuit. DEBUG ONLY.')
parser.add_argument('--problem', dest='problem', action='store',
                    default="problem0",
                    help='Name of the problem to test against.')
parser.add_argument('--sample_limit', "-n", action='store', type=int,
                    help='Number of training vectors to use - if your solution uses the hints, you can probably make this very small (~10) and train much more quickly.')

args = parser.parse_args()

fname = args.problem if "pyz" in args.problem else args.problem+".pyz"

with open(fname, "rb") as f:
    problem = pickle.load(f)

print("########## Problem hint: ####################")
print(problem["Hint"], end="")
if problem["Hint"][-1] != "\n":
    print()
print("########## Now running your code ############")

if args.print_problem_stats:
    print(f"number of training samples: {len(problem['TrainSamples'])}")
    print(f"label bias (sum/number): {sum(problem['TrainLabels']) / len(problem['TrainLabels'])}")
    print(f"Training ETA: {problem['TimeEst']}")

if args.sample_limit is None:
    sample_limit = min(len(problem['TrainSamples']), 20)
else:
    sample_limit = min(len(problem['TrainSamples']), args.sample_limit)

print(f"using {sample_limit} training examples out of {len(problem['TrainSamples'])}. change this with -n NUMBER.")

if args.cheat:
    from pprint import pprint
    pprint(problem)

if args.solution_function_name is None:
    print("Please provide the name of your proposed solution function as --fun [NAME] to evaluate. exiting.")
    sys.exit(0)

proposed_solution = trialmodule.__dict__[args.solution_function_name]

print(f"using {proposed_solution}")

t0 = time.time()
traindata = list(zip(problem["TrainSamples"], problem["TrainLabels"]))[:sample_limit]
traindata = list(traindata)
trained_result = proposed_solution( traindata )
dt = time.time() - t0

predictfn = trained_result["infer_fun"]

if not callable(predictfn):
    print("Your training function needs to return a dict from inference_retval!")
    sys.exit(0)


def getcost(fn, vectors, labels):
    acc = 0.0
    for vec, label in zip(vectors, labels):
        p = fn(vec)
        p = 1 if p>0 else -1 # round the result
        if (p == label):
            acc += 1

    acc *= 100/len(labels)

    return acc


training_accuracy = getcost(predictfn, problem["TrainSamples"][:sample_limit], problem["TrainLabels"][:sample_limit])
test_accuracy = getcost(predictfn, problem["TestVectors"], problem["TestLabels"])


#test_error = 0.0
#for testvec, testres in zip(problem["TestVectors"], problem["TestLabels"]):
#    p = predictfn(testvec)
#    test_error += abs(p-testres)
    # if abs(p-testres) > 0.0001:
    #     print(p, testres, testvec)
#accuracy_percentage = test_error/len(problem["TestVectors"]) * 100

## Now we have evaluated the users solution, we need to package up as much metadata
## as possible for later grading.

try:
    source = inspect.getsource(proposed_solution)
except Exception as e:
    print("failed to get source code for solution.")
    print(traceback.format_exc())
    source = None

circuit = trained_result["infer_circ"]
if circuit:
    circuit_str = print_circuit(circuit, num_qubits = int(np.log2(len(problem["TestVectors"][0]))) )
else:
    print("No circuit is available for this solution")
    circuit_str = None

problem_name    = args.problem
match_obj       = re.match('\D+(\d+)', problem_name)
problem_index   = int(match_obj.group(1)) if match_obj else -1

result_dict = {
    "problem_name":problem_name,
    "problem_index":problem_index,

    "training_vectors_limit":args.sample_limit,
    "solution_function_name":args.solution_function_name,
    "source_code":source,
    "circuit_str":circuit_str,
    "training_time":dt,
    "training_accuracy":training_accuracy,
    "test_accuracy":test_accuracy,
}


time_str = datetime.datetime.utcfromtimestamp(time.time()).strftime('%H:%M')

i = 0
while os.path.exists(f"{args.problem}_solution.{time_str}_{test_accuracy:.2f}_{i}.json"):
    i += 1
fname = f"{args.problem}_solution.{time_str}_{test_accuracy:.2f}_{i}.json"

with open(fname, "w") as f:
    json.dump(result_dict, f, indent=2)

print(f"Training accuracy: {training_accuracy:.2f}%, taking {dt:.1f} seconds to train. Test accuracy: {test_accuracy:.2f}%")

if dt > problem["TimeEst"]:
    print(f"It took more than {problem['TimeEst']} seconds to train your solution - we are sure there is a better method!")

print(f"Run saved to {fname}.\nUpload with:\n\tck store_experiment qml --json_file={fname}")
