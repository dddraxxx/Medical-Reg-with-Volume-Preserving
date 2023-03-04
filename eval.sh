
function check_m(){
    if [[ $msk_weight == *"normal"* ]]; then
        m=""
    else
        m=""
    fi
}
function single_eval(){
    msk_weight=$1
    shift
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $@
}
function eval() {
    msk_weight=$1
    check_m
    shift 
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 $@
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v lmk-val -lm -lm_r 10 $@
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v sliver-val -lm -lm_r 10 $@
}

function eval_lmr(){
    msk_weight=$1
    check_m
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v lmk-val -lm -lm_r $2
    # python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v sliver-val -lm -lm_r $2
}

function eval_sd() {
    msk_weight=$1
    check_m
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 1 -v lmk-val -lm -lm_r 10
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 1 -v sliver-val -lm -lm_r 10
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 1
}

function eval_vl(){
    msk_weight=$1
    check_m
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v lmk-val -lm -vl --name vl
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight $m -sd 0 -v sliver-val -lm -vl --name vl
}

msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/brain/VTN/1/Mar03-014518_br1_VTNx3_adaptive_softthr1.5sigmbnd0.5st1_vp0.1stdynamic
single_eval $msk_weight $@

