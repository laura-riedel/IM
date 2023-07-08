import argparse

###

parser = argparse.ArgumentParser(prog='RunVariableCNN',
                                description='Trains and tests a model configuration with variableCNN on UKBB data.',
                                epilog='Training + testing complete.')

parser.add_argument('-lr', type=float, default=1e-3, help='Enter learning rate for training. Default: 1e-3') #, metavar='learning rate'

args = parser.parse_args()

lr = args.lr

print(lr)