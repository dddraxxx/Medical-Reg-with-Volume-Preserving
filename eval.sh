# normal version
# python eval.py -d datasets/liver_cust.json -g 2 -c /home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180325_normal-vtn -sd 0 -v sliver -lm
# python eval.py -d datasets/liver_cust.json -g 2 -c /home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180325_normal-vtn -sd 0 -v lmk-val -lm -lm_r -1 -vl 
# msk version
# python eval.py -d datasets/liver_cust.json -g 2 -c /home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193345_msk-ks0.5-vtn -m -sd 0 -v sliver -lm --name ep10
# python eval.py -d datasets/liver_cust.json -g 2 -c /home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193345_msk-ks0.5-vtn -m -sd 1 --name ep10
# python eval.py -d datasets/liver_cust.json -g 2 -c /home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193345_msk-ks0.5-vtn/model_wts/epoch_4_iter_20000.pth -m -sd 0 -v lmk-val -lm -lm_r -1 -vl --name vl
# python eval.py -d datasets/liver_cust.json -g 2 -c /home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193345_msk-ks0.5-vtn/model_wts/epoch_4_iter_20000.pth -m -sd 1 -v lmk-val -lm -vl --name vl
# python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -m -sd 0 -v lmk-val -lm -lm_r 10
# python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -m -sd 0 -v sliver -lm -lm_r 10
# python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -m -sd 0

# python eval.py -d datasets/liver_cust.json -g 2 -c /home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193410_msk-ks1.5-vtn -m -sd 0 -v sliver -lm
# mskim version
# python eval.py -d datasets/liver_cust.json -g 2 -c /home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193622_msk-ks1.5im-vtn -m -sd 0 -v sliver -lm

# pre msk version
# python eval.py -d datasets/liver_cust.json -g 2 -c /home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180111_msk-ks1.5-vtn -m -sd 0 -v lmk-val -lm

msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan19_054605_msk-ks.1-vtn
function eval() {
    msk_weight=$1
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -m -sd 0 -v lmk-val -lm -lm_r 10
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -m -sd 0 -v sliver -lm -lm_r 10
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -m -sd 0
}

function eval_sd() {
    msk_weight=$1
    sd=$2
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -m -sd $sd -v lmk-val -lm -lm_r 10
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -m -sd $sd -v sliver -lm -lm_r 10
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -m -sd $sd
}

#regex: find dir with name ".*ks.\d.*"
for msk_weight in $(find $dir -type d -regex ".*ks\.[1-9].*"); do
    eval $msk_weight
done