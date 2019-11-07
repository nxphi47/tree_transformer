# tree_transformer
Submission to ICLR 2020: [https://openreview.net/forum?id=HJxK5pEYvr](https://openreview.net/forum?id=HJxK5pEYvr)

This is an unofficial example codes for IWSLT'14 En-De

# Installation

Install fairseq
```bash
# install the latest pytorch first
pip install --upgrade fairseq==0.6.2
pip install -U nltk[corenlp]

git clone https://github.com/XXXXX/tree_transformer.git
```

Install CoreNLP Stanford Parser [here](https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK)
Suppose the parser is stored in `stanford-corenlp-full-2018-02-27`

# Parsing and Preprocess translation data

Follow preparation of the data [here - Fairseq](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh).  
Suppose the data saved in `raw_data/iwslt14.tokenized.de-en.v2`, this contains the file train.en, train.de, valid.en, valid.de, test.en, test.de

Run CoreNLP server in a separate terminal
```bash
cd stanford-corenlp-full-2018-02-27/
port=9000
java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port $port -port $port -timeout 15000000 
```

Parse the data
```bash
# ----------------- German-English ------------------------

export PARSER_PORT=9000
export prefix=train

export cur=`pwd`
export root=${cur}/raw_data/iwslt14.tokenized.de-en.v2
export before=$root/$prefix.en
export after=$root/$prefix.tree-ende.en
export before_t=$root/$prefix.de
export after_t=$root/$prefix.tree-ende.de
export bpe=${root}/code
python -u tree_transformer/parse_nmt.py --ignore_error --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t}

# do the same for valud
# files train.tree-en.en, train.tree-ende.de, valid.tree-ende.en, valid.tree-ende.de, ....
```


Preprocess data into Fairseq
```bash
#   IWSLT - En-De
export ROOT_DIR=`pwd`
export PROJDIR=tree_transformer
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/iwslt14.tokenized.de-en.v2
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.tree-ende
export valid_r=${RAW_DIR}/train.tree-ende
export test_r=${RAW_DIR}/train.tree-ende
export OUT=${ROOT_DIR}/data_fairseq/nstack_merge_translate_ende_iwslt_32k
rm -rf $OUT
python -m tree_transformer.preprocess_nstack2seq_merge \
--source-lang en --target-lang de \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_remove_root \
--workers 8 \
--eval_workers 0 \

# processed data saved in data_fairseq/nstack_merge_translate_ende_iwslt_32k
``` 



# Training

```bash
export MAXTOKENS=1024
export INFER=y
export dis_port_str=--master_port=6102
export problem=nstack_merge_iwslt_ende_32k
export MAX_UPDATE=61000
export UPDATE_FREQ=1
export att_dropout=0.2
export DROPOUT=0.3 &&
bash run_nstack_nmt.sh dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier 0,1,2,3,4,5,6,7

```






