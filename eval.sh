
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

function eval_test(){
    msk_weight=$1
    shift
    python eval.py -d datasets/liver_cust.json -g 2 -c $msk_weight -sd 0 -v test  $@
}

##### VTN, Liver
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/brain/VTN/1/Mar03-014518_br1_VTNx3_adaptive_softthr1.5sigmbnd0.5st1_vp0.1stdynamic
# single_eval $msk_weight $@

# VTN
# msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/1/Mar05-153409_li1_VTNx3_adaptive-u2bf_softthr1.5sigmbnd0.5st2bf_vp0.1stdynamic

/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/1/Mar06-015754_li1_VTNx3_seg_seg_vp0.1sttumor
/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/1/Mar04-191415_li1_VTNx3_normal__
# eval_test $msk_weight $@

# msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/1/Mar07-063207_li1_VTNx3_random-seg0.2_seg_vp0.1sttumor
# msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/1/Mar07-041346_li1_VTNx3_random-seg0.01_seg_vp0.1sttumor
# /home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/1/Mar04-004325_li1_VTNx3_seg_seg_
/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/1/Mar04-233439_li1_VTNx3_unsup-organ_hard_vp0.1storgan


# VXM
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/liver/VXM/1/Mar01-191032_1_VXMx1_normal__
# eval_test $msk_weight $@
msk_weight=/home/hynx/regis/recursive-cascaded-networks/logs/liver/VXM/1/Mar08-070844_li1_VXMx1_adaptive_softthr1.5sigmbnd0.5st2bf_vp0.1stdynamic
# eval_test $msk_weight $@

# TSM
msk_weight=/workspace/qihuad/iccv23_regis/Recursive-Cascaded-Networks/logs/liver/TSM/1/Mar03-033707_li1_TSMx1_normal__
# msk_weight=/workspace/qihuad/iccv23_regis/Recursive-Cascaded-Networks/logs/liver/TSM/1/Mar05-030624_li1_TSMx1_adaptive-u2bf_softthr1.5sigmbnd0.5st2bf_vp0.1stdynamic
# msk_weight=/workspace/qihuad/iccv23_regis/Recursive-Cascaded-Networks/logs/liver/TSM/1/Mar06-023254_li1_TSMx1_seg_seg_vp0.1sttumor
eval_test $msk_weight $@


### brain
# normal VTN
/home/hynx/regis/recursive-cascaded-networks/logs/brain/VTN/1/Mar02-050938_1_VTNx3_normal___
# ours VTN
/home/hynx/regis/recursive-cascaded-networks/logs/brain/VTN/1/Mar03-014518_br1_VTNx3_adaptive_softthr1.5sigmbnd0.5st1_vp0.1stdynamic

/home/hynx/regis/recursive-cascaded-networks/logs/brain/VTN/1/Mar13-184519_br1_VTNx3_seg_seg_vp0.1sttumor
# normal VXM
/home/hynx/regis/recursive-cascaded-networks/logs/brain/VXM/1/Mar02-141247_1_VXMx1_normal__
# ours VXM
/home/hynx/regis/recursive-cascaded-networks/logs/brain/VXM/1/Mar08-061257_br1_VXMx1_seg_seg_vp0.1sttumor
# normal TSM
/workspace/qihuad/iccv23_regis/Recursive-Cascaded-Networks/logs/brain/TSM/1/Mar02-202418_br1_TSMx1_normal__
# ours TSM
/workspace/qihuad/iccv23_regis/Recursive-Cascaded-Networks/logs/brain/TSM/1/Mar05-161549_br1_TSMx1_adaptive-u2bf_softthr1.5sigmbnd0.5st2bf_vp0.1stdynamic