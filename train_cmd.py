#%% generate comands for training
base_command = 'python train.py'

import argparse
parser = argparse.ArgumentParser()
### On Brain or Liver
parser.add_argument('-d', '--dataset', type= lambda x: {
                        'brain': 'brain', 'liver': 'liver',
                        'b': 'brain', 'l': 'liver',
                    }[x.lower()]
                    , default='liver', help='brain or liver')
### Use Normal? Seg?  or adaptive? or hard?
parser.add_argument('-m', '--mode', type=lambda x: {
                        'normal': 'normal', 'seg': 'seg', 'adaptive': 'adaptive', 'organ': 'unsup-organ',
                        'n': 'normal', 's': 'seg', 'a': 'adaptive', 'o': 'unsup-organ',
                    }[x.lower()]
                    , default='', help='normal or seg or adaptive')
### Normal training or Mini Experiments
parser.add_argument('-t', '--type', type= lambda x: {
                        'normal': 'normal', 'mini': 'mini', 
                        'n': 'normal', 'm': 'mini',
                    }[x.lower()]
                    , default='normal', help='normal or mini')

#%%
args = parser.parse_args()
command = {}
if args.dataset == 'brain':
    command['-d'] = 'datasets/brain_cust.json'
elif args.dataset == 'liver':
    command['-d'] = 'datasets/liver_cust.json'

command['--name'] = args.mode
if args.mode == 'normal':
    pass
elif args.mode == 'seg':
    command['-m'] = 'seg'
elif args.mode == 'adaptive':
    command['-m'] = 'soft'
    command['-trsf'] = 'sigm'
    command['-use2'] = '0'
    command['-bnd_thick'] = '0.5'
    command['--mask_threshold'] = '1.5'
    command['-vp'] = '0.1'
    command['-st'] = 'dynamic'
elif args.mode == 'unsup-organ':
    command['-m'] = 'hard'
    command['--mask_threshold'] = '-1'
    command['-vp'] = '0.1'
    command['-st'] = 'organ'

if args.type == 'normal':
    pass
elif args.type == 'mini':
    command['-ts'] = 'mini'
    command['-e'] = '1'
    command['-r'] = '1000'
    command['-v'] = '-1'
    command['-cf'] = '1'

command = base_command + ' ' + ' '.join([f'{k} {v}' for k, v in command.items()])
new_args = input('Please input extra args and press Enter to run the following command: \n' + command)
import os
print()
command = command + ' ' + new_args
print('Running command: \n' + command)
os.system(command)

# python train_cmd.py -d b -t m -m a