#! python3
import argparse
import pickle
import sys
import time
import small_circuits as trialmodule

parser = argparse.ArgumentParser(description='Tests your solutions for the quantum classification problem.')
parser.add_argument('--solution_function_name', "--fun", metavar='S', type=str,
                    help='The name of your function in small_circuits.py')
parser.add_argument('--print_problem_stats', "--stats", action='store_true',
                    help='Prints some statistics about the training data and the problem.')
parser.add_argument('--cheat', action='store_true',
                    help='Prints the transformation circuit. DEBUG ONLY.')
parser.add_argument('--problem', dest='problem', action='store',
                    default="problem3",
                    help='Name of the problem to test against.')
parser.add_argument('--sample_limit', "--n", action='store',
                    default="-1", type=int,
                    help='Number of training vectors to use - if your solution uses the hints, you can probably make this very small (~10) and train much more quickly.')

args = parser.parse_args()

with open(args.problem+"_spec.pyz", "rb") as f:
    problem = pickle.load(f)

print("")
print(problem["Hint"], end="")

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
predictfn = proposed_solution( traindata )
dt = time.time() - t0

if not callable(predictfn):
    print("Your training function needs to return a callable classification function!")
    sys.exit(0)

cost = 0.0
for testvec, testres in zip(problem["TestVectors"], problem["TestLabels"]):
    p = predictfn(testvec)
    cost += abs(p-testres)
    # if abs(p-testres) > 0.0001:
    #     print(p, testres, testvec)

print(f"error in your solution was {cost:.5f}, taking {dt:.2e} seconds to train.")
if dt > problem["TimeEst"]:
    print(f"It took more than {problem['TimeEst']} to train your solution - we are sure there is a better method!")
