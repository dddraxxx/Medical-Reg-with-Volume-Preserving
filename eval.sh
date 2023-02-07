
function check_m(){
    if [[ $msk_weight == *"normal"* ]]; then
        m=""
    else
        m="-m"
    fi
}
function single_eval(){
    msk_weight=$1
    check_m
    shift
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m $@
}
function eval() {
    msk_weight=$1
    check_m
    shift 
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v lmk-val -lm -lm_r 10 $@
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v sliver -lm -lm_r 10 $@
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 $@
}

function eval_lmr(){
    msk_weight=$1
    check_m
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v lmk-val -lm -lm_r $2
    # python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v sliver -lm -lm_r $2
}

function eval_sd() {
    msk_weight=$1
    check_m
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 1 -v lmk-val -lm -lm_r 10
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 1 -v sliver -lm -lm_r 10
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 1
}

function eval_vl(){
    msk_weight=$1
    check_m
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v lmk-val -lm -vl --name vl
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v sliver -lm -vl --name vl
}
# normal version
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180325_normal-vtn
# eval $msk_weight

# msk version
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193345_msk-ks0.5-vtn
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193345_msk-ks0.5-vtn/model_wts/epoch_4_iter_20000.pth
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193410_msk-ks1.5-vtn

# mskim version
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan17_193622_msk-ks1.5im-vtn

# pre msk version
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180111_msk-ks1.5-vtn

msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan19_054605_msk-ks.1-vtn
# eval $msk_weight
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan27_182001_msk-ks.15-vtn

# test a new model
# msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan31_163313_msk-wholeks.1-vtn
# msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Feb02_002511_hard-wholeks.1-vtn
# eval $msk_weight -mt hard
# eval_lmr $msk_weight 10

# eval_lmr $msk_weight 10
#regex: find dir with name ".*ks.\d.*"
# for msk_weight in $(find ./logs  -maxdepth 1 -type d -regex ".*ks\.[1-9]-.*"); do
#     eval_lmr $msk_weight -1
# done


# test generated dataset lits_p
# msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180325_normal-vtn
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Feb02_002511_hard-wholeks.1-vtn
single_eval $msk_weight -v lits-p -lm -sd 0 -lm_r 10
# msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/Jan19_054605_msk-ks.1-vtn
# single_eval $msk_weight -v lits-p -lm -sd 0 -lm_r 10