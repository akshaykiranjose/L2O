from utils import *
from quadratic import Quadratic, QuadraticLoss
from model import LSTMOpt
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description = "Run Learned Optimizer comparison with other optimizers")
parser.add_argument('-s', '--seed', type=int, required=True, help = 'random seed for reproducing results' )
parser.add_argument('-nF', '--numFuncs', type=int, required = True, help = 'number of functions to evaluate on' )
parser.add_argument('-nI', '--numIters', type=int, required = True, help = 'number of iterations to optimize each function for')
parser.add_argument('-lr', '--learnRate', type=float, required = True, help = 'learning rate for mathematical optimizers')
parser.add_argument('-p', '--plot', type=bool, required = True, help = 'plot results or not')
args = parser.parse_args()


if __name__ == "__main__":

	physical_devices = tf.config.list_physical_devices('GPU')
	if(len(physical_devices) > 0):
		print("Yes GPU")
	else:
		print("No GPU")

	lstm = LSTMOpt()
	Fit_learned(lstm, Quadratic, QuadraticLoss)

	Evaluate(lstm, Quadratic, args.seed, args.numFuncs, args.numIters, args.learnRate, args.plot)