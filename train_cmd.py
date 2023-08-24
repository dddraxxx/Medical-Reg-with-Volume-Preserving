#%% generate comands for training
base_command = 'python train_simple.py'

import argparse
parser = argparse.ArgumentParser()
### get name
parser.add_argument('-n', '--name', type=str, default='', help='name of the experiment')
### On Brain or Liver
parser.add_argument('-d', '--dataset', type= lambda x: {
                        'brain': 'brain', 'liver': 'liver',
                        'b': 'brain', 'l': 'liver',
                    }[x.lower()]
                    , default='liver', help='brain or liver')
### Use Normal? Seg?  or adaptive? or hard?
rsv=-1
hv =2
def mode_(x):
    dct = {'normal': 'normal', 'seg': 'seg', 'adaptive': 'adaptive', 'organ': 'unsup-organ', 'tumor':'unsup-tumor', 'random-seg': 'random-seg', 'hard': 'hard',
                        'n': 'normal', 's': 'seg', 'a': 'adaptive', 'o': 'unsup-organ', 't': 'unsup-tumor', 'rs': 'random-seg', 'h': 'hard',
                    }
    if x.lower() in dct: return dct[x.lower()]
    if x.lower().startswith('rs-') or x.lower().startswith('random-seg'):
        rsv = float(x.split('-')[-1])
        globals()['rsv'] = rsv
        return f'random-seg{rsv}'
    elif x.lower().startswith('h-') or x.lower().startswith('hard'):
        hv = float(x.split('-')[-1])
        globals()['hv'] = hv
        return f'hard{hv}'
parser.add_argument('-m', '--mode', type=lambda x:mode_(x)
                    , default='', help='normal or seg or adaptive')
### Normal training or Mini Experiments
parser.add_argument('-t', '--type', type= lambda x: {
                        'normal': 'normal', 'mini': 'mini',
                        'n': 'normal', 'm': 'mini',
                    }[x.lower()]
                    , default='normal', help='normal or mini')
### Base network: VTN, VXM or DMR, TSM
parser.add_argument('-b', '--base', type= lambda x: {
                        'vtn': 'vtn', 'vxm': 'vxm', 'dmr': 'dmr', 'tsm': 'tsm',
                        'v': 'vtn', 'x': 'vxm', 'd': 'dmr', 't': 'tsm',
                    }[x.lower()]
                    , default='vtn', help='vtn or vxm or dmr')
### add debug flag
parser.add_argument('-db', '--debug', action='store_true', help='debug mode')
# hyper net
parser.add_argument('-hyp', action="store_true", help="whether use hypernet")
#%%
args = parser.parse_args()
command = {}
if args.dataset == 'brain':
    command['-d'] = 'datasets/brain_cust.json'
elif args.dataset == 'liver':
    command['-d'] = 'datasets/liver_cust.json'

command['--name'] = args.mode+ (args.name and '-'+args.name)
if args.mode == 'normal':
    pass
elif args.mode == 'seg':
    command['-m'] = 'seg'
    command['-vp'] = '0.1'
    command['-st'] = 'tumor'
elif args.mode.startswith('random-seg'):
    command['-m'] = 'seg'
    command['-vp'] = '0.1'
    command['-st'] = 'tumor'
    command['-msd'] = f'{rsv}'
elif args.mode == 'adaptive':
    command['-m'] = 'soft'
    command['-trsf'] = 'sigm'
    command['-use2'] = '1'
    command['-bnd_thick'] = '0.5'
    command['--mask_threshold'] = '1.5'
    command['-vp'] = '0.1'
    command['-st'] = 'dynamic'
elif args.mode.startswith('hard'):
    command['-m'] = 'hard'
    command['--mask_threshold'] = f'{hv}'
    command['-bnd_thick'] = '0.5'
    command['-vp'] = '0.1'
    command['-st'] = 'tumor'
elif args.mode == 'unsup-organ':
    command['-m'] = 'hard'
    command['--mask_threshold'] = '-1'
    command['-vp'] = '0.1'
    command['-st'] = 'organ'
elif args.mode == 'unsup-tumor':
    command['-m'] = 'hard'
    command['--mask_threshold'] = '2'
    command['-vp'] = '0.1'
    command['-st'] = 'tumor'

if args.type == 'normal':
    pass
elif args.type == 'mini':
    command['-ts'] = 'midi'
    command['-e'] = '1'
    command['-r'] = '2000'
    command['-v'] = '-1'
    command['-cf'] = '1'

if args.base == 'vtn':
    command['-base'] = 'VTN'
elif args.base == 'vxm':
    command['-base'] = 'VXM'
elif args.base == 'dmr':
    command['-base'] = 'DMR'
    command['-ua'] = '0'
elif args.base == 'tsm':
    command['-base'] = 'TSM'

if args.hyp:
    command['-hpv'] = ''

if args.debug:
    command['--debug'] = ''

command['-wb'] = 0

command = base_command + ' ' + ' '.join([f'{k} {v}' for k, v in command.items()])
new_args = input('Please input extra args and press Enter to run the following command: \n' + command)
import os
print()
command = command + ' ' + new_args
print('Running command: \n' + command)
os.system(command)

# python train_cmd.py -d b -t m -m a