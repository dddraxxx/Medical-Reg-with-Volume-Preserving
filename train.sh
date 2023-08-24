# continue training on
python train.py --ctt /home/hynx/regis/recursive-cascaded-networks/logs/Feb26_021930_1-softsig1.5*5st1bnd0-vpdy.1 -m soft -vp 0.1 -st dynamic -bnd_thick 0 -use2 0 -trsf sigm

# used to extract stage1 tumors
python stage1.py -g 3 --name stage1 -m hard -vp 0.1 -st organ --debug -ts mini

# regular training
python train.py -g 2 --name softsig1.5*5-vpdy.1 -m soft -vp 0.1 -st dynamic -trsf sigm -bnd_thick 1
python train.py -g 4 --name softsin1.5-vpdy.1 -m soft -vp 0.1 -st dynamic -trsf sin -bnd_thick 1
python train.py -g 2 --name softsig1.5*5-vpdy.1 -m soft -vp 0.1 -st dynamic -trsf sigm
python train.py -g 2 --name softsig1.5*5st1bnd0-vpdy.1 -m soft -vp 0.1 -st dynamic -trsf sigm -use2 0 -bnd_thick 0
python train.py -g 2 --name softsig1.5*5st1bnd0-vpdy.1 -m soft -vp 0.1 -st dynamic -trsf sigm -use2 0 -bnd_thick 0 -base VXM
# use precompute
python train.py -g 2 --name 1pc-softsig1.5*5st1bnd0-vpdy.1 -m soft -vp 0.1 -st dynamic -trsf sigm -use2 0 -bnd_thick 0 -pc 1

python train.py -g 3 --name 1-hard-vporgan.1 -m hard -vp 0.1 -st organ
python train.py -g 3 --name 1-hard-vporgan.1 -m hard -vp 0.1 -st organ -base VXM --debug
python train.py -g 4 --name 1-seg-vpwarpedtumor.1 -m seg -vp 0.1 --base VXM
python train.py -g 5 --name 1-normal --debug
# training on brain dataset
python train.py --name br-normal -d datasets/brain_cust.json --debug
python train.py --name br-seg -d datasets/brain_cust.json -m seg -vp .1 -e 1 -ts mini -r 3000
python train.py --name br-soft -d datasets/brain_cust.json -m soft -vp 0.1 -st dynamic -trsf sigm -use2 0 -bnd_thick 0 --debug
python train.py --name br-softsig1.5x5st1bnd0-vpdy.1 -d datasets/brain_cust.json -m soft -vp 0.1 -st dynamic -trsf sigm -use2 0 -bnd_thick 0 -base VXM

## integrated with stage1
python train.py -g 3 --name 1-hard-vptumor.1 -m hard -vp 0.1 -st tumor

# train with seg mask
python train.py -g 4 --name mini-seg-ks.1 -m seg -vp 0.1 -ts mini -cf 0.5 -e 1 --lr 3e-4 -v -1 -r 3000

#-----------------
##### mini experiments
# train with computed hard mask
python train.py -g 2 --name mini1451-soft-vpdyna.1 -m soft -vp 0.1 -st dynamic -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
python train.py -g 3 --name mini1451-soft-vpdyna.2 -m soft -vp 0.2 -st dynamic -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
python train.py -g 2 --name mini1451-soft-vpdyna.1-sigm -m soft -vp 0.1 -st dynamic -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1 --debug
python train.py -g 4 --name mini1451-hard-vpdtum.1 -m hard -vp 0.1 -st tumor -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
python train.py -g 2 --name mini1451-soft-vpdyna.2-sigm -m soft -vp 0.2 -st dynamic -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1

python train.py -g 4 --name mini1451-softsig1.5*5-vpdy.1 -m soft -vp 0.1 -st dynamic -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
python train.py -g 4 --name mini1451-softsig1.5*5bnd.5-vpdy.1 -m soft -vp 0.1 -st dynamic -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
python train.py -g 4 --name mini1451-softsig1.5*51st-vpdy.1 -m soft -vp 0.1 -st dynamic -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1 -use2 0 -bnd_thick 1


python train.py -g 4 --name mini5133-softsig1.5*5bnd0-vpdy.1 -m soft -vp 0.1 -st dynamic -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1 -bnd_thick 0

python train.py -g 1 --name mini1451-softsin1.5-vpdy.1 -m soft -vp 0.1 -st dynamic -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1 -trsf sin
python train.py -g 1 --name mini1451-softsin1.5bnd.5-vpdy.1 -m soft -vp 0.1 -st dynamic -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1 -trsf sin

python train.py -g 4 --name mini5133-softsig1.5*1-vpdy.1 -m soft -vp 0.1 -st dynamic -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
python train.py -g 3 --name mini5133-softsig1.5*5-vpdy.1 -m soft -vp 0.1 -st dynamic -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
python train.py -g 1 --name mini5133-softsin1.5-vpdy.1 -m soft -vp 0.1 -st dynamic -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1 -trsf sin
python train.py -g 5 --name mini5133-seg-tumorvp.1 -m seg -vp 0.1 -st tumor -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
python train.py -g 3 --name mini5133-hard-organvp.1 -m hard -vp 0.1 -st organ -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1


python train.py -g 2 --name mini-hard-vptumor.1 -m hard -vp 0.1 -st tumor -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000
python train.py -g 3 --name mini-wholeks.1-kmskratio -m hard -msk_thr 10 -vp 0.1 -st whole -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000

python train.py -g 3 --name mini-wholeks.1-kmskratio -m hard -msk_thr 10 -vp 0.1 -st whole -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000

python train.py -g 5 --name mini-hard-wholeks.1 -m hard -vp 0.1 -st whole -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1

python train.py -g 4 --name mini-hard-organks.1 -m hard -vp 0.1 -st organ -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000

python train.py -g 4 --name mini-reg+organks.1 -m hard -vp 0.1 -msk_thr 10 -st organ -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 --reg 1.1 -e 1

python train.py -g 3 --name mini-normal -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1

python train.py -g 2 --name mini-organvp.1 -vp 0.1 -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1

python train.py -g 5 --name mini1451-seg-tumorvp.1 -m seg -vp 0.1 -st tumor -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
python train.py -g 2 --name mini1451-hard-organvp.1 -m hard -vp 0.1 -st organ -ts mini-hard -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1


# python train.py -g 4 --name mini-hard -m hard -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
# python train.py -g 1 --name mini-hard-reg.1 -m hard -vp 0.1 -st reg -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
# python train.py -g 2 --name mini-hard-wholeks.2 -m hard -vp 0.2 --reg 0.9 -st whole -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1
# python train.py -g 2 --name mini-hard -m hard -st whole -ts mini -cf 0.5 --lr 3e-4 -v -1 -r 3000 -e 1


