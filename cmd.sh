# normal hyp vxm
python train_simple.py -d datasets/liver_cust.json --name normal-hypx -base VXM -wb 0 -hpv --lr 2e-4

# hyp vp vxm
python train_simple.py -d datasets/liver_cust.json --name adaptive-vp-hypx -m soft -trsf sigm -use2 1 -bnd_thick 0.5 --mask_threshold 1.5 -vp 0.1 -st dynamic -base VXM -hpv  -wb 0 --lr 2e-4