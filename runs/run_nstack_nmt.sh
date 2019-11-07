#!/usr/bin/env bash


set -e

export arch=$1
export gpus=$2


export CUDA_VISIBLE_DEVICES=$gpus
export id="${id:-50}"
export fp16="${fp16:-0}"


#export problem="${problem:-nstack_iwslt_ende_v2}"
export problem="${problem:-nstack_merge_iwslt_ende_32k}"

export RM_EXIST_DIR="${RM_EXIST_DIR:-y}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, problem=${problem}"



export dis_port_str="${dis_port_str:-}"

export KEEP_LAS_CHECKPOINT="${KEEP_LAS_CHECKPOINT:-20}"


#if [ ${problem} == "nstack_merge_iwslt_ende_32k" ]; then
export PROBLEM=nstack_merge_translate_ende_iwslt_32k
export TASK=nstack_merge2seq

export HPARAMS=transformer_base
export LEFT_PAD_SRC=False

#    export MAXTOKENS=2048
export MAXTOKENS="${MAXTOKENS:-1024}"
export UPDATE_FREQ="${UPDATE_FREQ:-1}"
export RM_EXIST_DIR="${RM_EXIST_DIR:-n}"

export INFER="${INFER:-y}"
export max_pos="${max_pos:-1024}"
export max_tgt_pos="${max_tgt_pos:-1024}"
export more_params="${more_params:-}"
# export more_params=--on_filter_nsent

#    export extra_params="--append-eos-to-target --max-source-positions ${max_pos} --max-target-positions ${max_tgt_pos}"
export extra_params="--append-eos-to-target ${more_params} --max-source-positions ${max_pos} --max-target-positions ${max_tgt_pos}"

export ID=${id}msp${max_pos}default
export DDP_BACKEND="${DDP_BACKEND:-no_c10d}"

export out_log="${out_log:-y}"
export log_dir=`pwd`/../../${problem}-logs/
#export log_dir=`pwd`/../../gpu4-logs/
mkdir -p ${log_dir}
export log_file=${log_dir}/${PROBLEM}-${arch}-${ID}.log
#    export MAX_UPDATE="${MAX_UPDATE:-35500}"
export MAX_UPDATE="${MAX_UPDATE:-45000}"
export nobar=1

export LR="${LR:-0.0005}"
export DROPOUT="${DROPOUT:-0.3}"
export WDECAY="${WDECAY:-0.0001}"
export AVG_NUM=10
export LENPEN=1

export ARCH=${arch}



bash train_fairseq.sh