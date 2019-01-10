#! /usr/bin/env python3
import argparse
import pickle
import sys
import os
import time
import datetime
import json
import inspect

import example_solutions as trialmodule

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
parser.add_argument('--sample_limit', "--n", action='store',
                    default="-1", type=int,
                    help='Number of training vectors to use - if your solution uses the hints, you can probably make this very small (~10) and train much more quickly.')

args = parser.parse_args()

with open(args.problem+"_spec.pyz", "rb") as f:
    problem = pickle.load(f)

print("########## Problem hint: ####################")
print(problem["Hint"], end="")
print("########## Now running your code ############")

if args.print_problem_stats:
    print(f"number of training samples: {len(problem['TrainSamples'])}")
    print(f"label bias (sum/number): {sum(problem['TrainLabels']) / len(problem['TrainLabels'])}")
    print(f"Training ETA: {problem['TimeEst']}")

if args.cheat:
    print("Problem circuit:")
    for gate, bit in problem['U']:
        print(str(gate), "|", bit)

if args.solution_function_name is None:
    print("Please provide the name of your proposed solution function as --fun [NAME] to evaluate. exiting.")
    sys.exit(0)

proposed_solution = trialmodule.__dict__[args.solution_function_name]

print(f"using {proposed_solution}")

t0 = time.time()
traindata = zip(problem["TrainSamples"], problem["TrainLabels"]) if args.sample_limit < 0 else \
            list(zip(problem["TrainSamples"], problem["TrainLabels"]))[:args.sample_limit]
traindata = list(traindata)
trained_result = proposed_solution( traindata )
dt = time.time() - t0

predictfn = trained_result["infer_fun"]

if not callable(predictfn):
    print("Your training function needs to return a dict from inferance_retval!")
    sys.exit(0)

cost = 0.0
for testvec, testres in zip(problem["TestVectors"], problem["TestLabels"]):
    p = predictfn(testvec)
    cost += abs(p-testres)
    # if abs(p-testres) > 0.0001:
    #     print(p, testres, testvec)
accuracy_percentage = cost/len(problem["TestVectors"]) * 100

## Now we have evaluated the users solution, we need to package up as much metadata
## as possible for later grading.

try:
    source = inspect.getsource(proposed_solution)
except exception as e:
    print("failed to get source code for solution!")
    print(traceback.format_exc())
    source = None

result_dict = {
    "problem":args.problem,
    "cost":cost,
    "circuit":str(trained_result["infer_circ"]),
    "test_accuracy":accuracy_percentage,
    "source_code":source
}


time_str = datetime.datetime.utcfromtimestamp(time.time()).strftime('%H:%M')
accuracy_str = f"{accuracy_percentage}"

i = 0
while os.path.exists(f"{args.problem}_solution.{time_str}_{accuracy_percentage:.2f}_{i}.json"):
    i += 1
fname = f"{args.problem}_solution.{time_str}_{accuracy_percentage:.2f}_{i}.json"

with open(fname, "w") as f:
    json.dump(result_dict, f, indent=2)

print(f"Error in your solution was {cost:.5f}, taking {dt:.2e} seconds to train.")

if dt > problem["TimeEst"]:
    print(f"It took more than {problem['TimeEst']} seconds to train your solution - we are sure there is a better method!")

print(f"Run saved to {fname}. Upload with ck upload ...")
