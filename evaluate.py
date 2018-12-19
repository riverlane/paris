import argparse
import pickle
import sys
import time
import small_circuits as trialmodule


parser = argparse.ArgumentParser(description='Tests your solutions for the quantum classification problem.')
parser.add_argument('--solution_name', metavar='S', type=str,
                    help='The name of your function in small_circuits.py')
parser.add_argument('--print_problem_stats', action='store_true',
                    help='Prints some statistics about the training data and the problem.')
parser.add_argument('--problem', dest='problem', action='store',
                    default="problem3",
                    help='Name of the problem to test against.')

args = parser.parse_args()

with open(args.problem+"_spec.pyz", "rb") as f:
    problem = pickle.load(f)

print("")
print(problem["Hint"], end="")

if args.print_problem_stats:
    print(f"number of training samples: {len(problem['TrainSamples'])}")
    print(f"label bias (sum/number): {sum(problem['TrainLabels']) / len(problem['TrainLabels'])}")
    print(f"Training ETA: {problem['TimeEst']}")



if args.solution_name is None:
    print("Please provide the name of your proposed solution function as --solution_name [NAME] to evaluate. exiting.")
    sys.exit(0)

proposed_solution = trialmodule.__dict__[args.solution_name]

print(f"using {proposed_solution}")

t0 = time.time()
predictfn = proposed_solution( zip(problem["TrainSamples"], problem["TrainLabels"]) )
dt = time.time() - t0

cost = 0.0
for testvec, testres in zip(problem["TestVectors"], problem["TestLabels"]):
    p = predict(testvec)
    cost += abs(p-testres)

print(f"error in your solution was {cost}, taking {dt}s to train.")
if dt > problem["TimeEst"]:
    print(f"It took more than {problem['TimeEst']} to train your solution - we are sure there is a better method!")
