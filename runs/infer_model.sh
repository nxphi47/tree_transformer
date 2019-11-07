#!/usr/bin/env bash



set -e
# specify machines

[ -z "$CUDA_VISIBLE_DEVICES" ] && { echo "Must set export CUDA_VISIBLE_DEVICES="; exit 1; } || echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPU=${#GPUS[@]}


export MACHINE="${MACHINE:-ntu}"

echo "MACHINE -> ${MACHINE}"

export ROOT_DIR=`pwd`
export PROJDIR=tree_transformer
export ROOT_DIR="${ROOT_DIR/\/tree_transformer\/runs/}"


export user_dir=${ROOT_DIR}/${PROJDIR}



if [ -d ${TRAIN_DIR} ]; then
	# if train exists
    echo "directory train exists!: ${TRAIN_DIR}"
else
    echo "directory train not exists!: ${TRAIN_DIR}"
    exit 1
fi


#export EPOCHS="${EPOCHS:-300}"
export PROBLEM="${PROBLEM:-nstack_merge_translate_ende_iwslt_32k}"

export RAW_DATA_DIR=${ROOT_DIR}/raw_data_fairseq/${PROBLEM}
export DATA_DIR=${ROOT_DIR}/data_fairseq/${PROBLEM}
export TRAIN_DIR_PREFIX=${ROOT_DIR}/train_tree_transformer/${PROBLEM}

export EXP="${EXP:-transformer_wmt_ende_8gpu1}"
export ID="${ID:-1}"
export INFER_ID="${INFER_ID:-1}"

export extra_params="${extra_params:-}"

export TGT_LANG="${TGT_LANG:-de}"
export SRC_LANG="${SRC_LANG:-en}"
export TESTSET="${TESTSET:-newstest2014}"


export INFERMODE="${INFERMODE:-avg}"
export INFER_DIR=${TRAIN_DIR}/infer
mkdir -p ${INFER_DIR}

# generate parameters

export TASK="${TASK:-translation}"
export BEAM="${BEAM:-5}"
#export INFER_BSZ="${INFER_BSZ:-128}"
#export INFER_BSZ="${INFER_BSZ:-4096}"
export INFER_BSZ="${INFER_BSZ:-2048}"
export LENPEN="${LENPEN:-0.6}"
#export LEFT_PAD_SRC="${LEFT_PAD_SRC:-True}"
export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"
export RMBPE="${RMBPE:-y}"
export GETBLEU="${GETBLEU:-y}"
export NEWCODE="${NEWCODE:-y}"
export INFER_TASK="${INFER_TASK:-mt}"
export HEAT="${HEAT:-n}"
export rm_srceos="${rm_srceos:-0}"
export rm_lastpunct="${rm_lastpunct:-0}"
export get_entropies="${get_entropies:-0}"
export GENSET="${GENSET:-test}"

export GEN_DIR=${INFER_DIR}/${GENSET}.tok.rmBpe${RMBPE}.genout.${TGT_LANG}.b${BEAM}.lenpen${LENPEN}.leftpad${LEFT_PAD_SRC}.${INFERMODE}


[ ${rm_srceos} -eq 1 ] && export rm_srceos_s="--remove-eos-from-source " || export rm_srceos_s=
[ ${rm_lastpunct} -eq 1 ] && export rm_lastpunct_s="--remove-last-punct-source " || export rm_lastpunct_s=
[ ${get_entropies} -eq 1 ] && export get_entropies_s="--layer-att-entropy " || export get_entropies_s=
[ ${RMBPE} == "y" ] && export rm_bpe_s="--remove-bpe " || export rm_bpe_s=

#/projects/nmt/train_tree_transformer/wmt16_en_de_new_bpe/me_vaswani_wmt_en_de_big-transformer_big_128-b5120-gpu8-upfre16-1fp16-id24
echo "========== INFERENCE ================="
echo "TASK = ${TASK}"
echo "infermode = ${INFERMODE}"
echo "BEAM = ${BEAM}"
echo "INFER_BSZ = ${INFER_BSZ}"
echo "LENPEN = ${LENPEN}"
echo "LEFT_PAD_SRC = ${LEFT_PAD_SRC}"
echo "RMBPE = ${RMBPE}"
echo "GETBLEU = ${GETBLEU}"
echo "NEWCODE = ${NEWCODE}"
echo "rm_srceos = ${rm_srceos} - string=${rm_srceos_s}"
echo "rm_lastpunct = ${rm_lastpunct} - string=${rm_lastpunct_s}"
echo "========== INFERENCE ================="

# selecting infermode
# ---------------------------------------------------------------------------------------------------
if [ ${INFERMODE} == "best" ]; then

    export CHECKPOINT=${TRAIN_DIR}/checkpoint_best.pt
    mkdir -p ${GEN_DIR}

    export GEN_OUT=${GEN_DIR}/infer
    export HYPO=${GEN_OUT}.hypo
    export REF=${GEN_OUT}.ref
    export BLEU_OUT=${GEN_OUT}.bleu

    echo "GEN_OUT = ${GEN_OUT}"

# ---------------------------------------------------------------------------------------------------------
elif [ ${INFERMODE} == "avg" ]; then

    export AVG_NUM="${AVG_NUM:-5}"
#    export UPPERBOUND="${UPPERBOUND:-22}"
    export UPPERBOUND="${UPPERBOUND:-100000000}"
#    export AVG_CHECKPOINT_OUT="${AVG_CHECKPOINT_OUT:-$TRAIN_DIR/averaged_model.${AVG_NUM}.u${UPPERBOUND}.pt}"
    export LAST_EPOCH=`python get_last_checkpoint.py --dir=${TRAIN_DIR}`

    export GEN_DIR=${GEN_DIR}.avg${AVG_NUM}.e${LAST_EPOCH}.u${UPPERBOUND}
    mkdir -p ${GEN_DIR}

    export AVG_CHECKPOINT_OUT="${AVG_CHECKPOINT_OUT:-$GEN_DIR/averaged_model.id${INFER_ID}.avg${AVG_NUM}.e${LAST_EPOCH}.u${UPPERBOUND}.pt}"
    export GEN_OUT=${GEN_DIR}/infer
    export GEN_OUT=${GEN_OUT}.avg${AVG_NUM}.b${BEAM}.lp${LENPEN}
    export HYPO=${GEN_OUT}.hypo
    export REF=${GEN_OUT}.ref
    export BLEU_OUT=${GEN_OUT}.bleu

    echo "GEN_DIR = ${GEN_DIR}"
    echo "GEN_OUT = ${GEN_OUT}"


    echo "AVG_NUM = ${AVG_NUM}"
    echo "LAST_EPOCH = ${LAST_EPOCH}"
    echo "AVG_CHECKPOINT_OUT = ${AVG_CHECKPOINT_OUT}"
    echo "---- Score by averaging last checkpoints ${AVG_NUM} -> ${AVG_CHECKPOINT_OUT}"
    echo "Generating average checkpoints..."
#    exit 1

    if [ -f ${AVG_CHECKPOINT_OUT} ]; then
        echo "File ${AVG_CHECKPOINT_OUT} exists...."
    else
        python ../scripts/average_checkpoints.py \
        --user-dir ${user_dir} \
        --inputs ${TRAIN_DIR} \
        --num-epoch-checkpoints ${AVG_NUM} \
        --checkpoint-upper-bound ${UPPERBOUND} \
        --output ${AVG_CHECKPOINT_OUT}
        echo "Finish generating averaged, start generating samples"
    fi

    export CHECKPOINT=${AVG_CHECKPOINT_OUT}

else
	echo "INFERMODE invalid: ${INFERMODE}"
	exit 1
fi


echo "Start generating"



export command="$(which fairseq-generate) ${DATA_DIR} \
    --task ${TASK} \
    --user-dir ${user_dir} \
    --path ${CHECKPOINT} \
    --left-pad-source ${LEFT_PAD_SRC} \
    --max-tokens ${INFER_BSZ} \
    --beam ${BEAM} \
    --gen-subset ${GENSET} \
    --lenpen ${LENPEN} \
    ${extra_params} \
    ${rm_bpe_s} ${rm_srceos_s} ${rm_lastpunct_s} | dd  of=${GEN_OUT}"
#                ${rm_bpe_s} ${rm_srceos_s} ${rm_lastpunct_s} | tee ${GEN_OUT}"

echo "Command: ${command}"
echo "----------------------------------------"
eval ${command}


echo "---- Score by score.py for mode=${INFERMODE}, avg=${AVG_NUM} -----"
echo "decode bleu from model ${AVG_CHECKPOINT_OUT}"
echo "decode bleu from file ${GEN_OUT}"
echo ".............................."

export SRC=${GEN_OUT}.src
export HYPO=${GEN_OUT}.hypo
export REF=${GEN_OUT}.ref
export REF_TW=${GEN_OUT}.ref.tweak
export BLEU_OUT=${GEN_OUT}.bleu

grep ^S ${GEN_OUT} | cut -f2- > ${SRC}
grep ^T ${GEN_OUT} | cut -f2- > ${REF}
grep ^H ${GEN_OUT} | cut -f3- > ${HYPO}



$(which fairseq-score) --sys ${HYPO} --ref ${REF} > ${BLEU_OUT}
cat ${BLEU_OUT}
echo ""


