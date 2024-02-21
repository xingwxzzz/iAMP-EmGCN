# iAMP-EmGCN
iAMP-EmGCN AMP classifier

## Introduction

A new deep learning method named iAMP-EmGCN, which combines RoBERTa and GCN, is proposed to predict the AMPs in this work. What calls for special attention is that this new design is based on BERTGCN method proposed by Lin et al, which has been improved to become the AMPs predictor we need. As far as we know, this is the first work to use RoBERTa and GCN methods to establish an identification model of AMPs. AMPs and non-AMPs datasets are preprocessed firstly, and then a heterogeneous graph is constructed based on word co-occurrence technology and RoBERTa word embedding technology in our work. Finally, GCN and RoBERTa model are used to train and predict AMPs to obtain high prediction accuracy of AMPs. The results show that our method can achieve better performance than other state-of-the-art methods proposed currently across the dataset used in this paper.



## Usage

1. Run `python preprocess.py` to combine positive and negative samples.
2. Run `python remove_words.py AMPs` to preprocess data.
3. Run `python build_graph.py AMPs` to build the AMP graph.
4. Run `python train_bert_gcn.py --dataset AMPs --pretrained_bert_ckpt [pretrained_bert_ckpt] -m [m]`
to train the iAMP-EmGCN. 
`[m]` is the factor balancing BERT and GCN prediction. 
The model and training logs will be saved to `checkpoint/[bert_init]_[gcn_model]_[dataset]/` by default. 
Run `python train_bert_gcn.py -h` to see the full list of hyperparameters.



## Acknowledgement

The original data contributions presented in the study are included in the directory "/data/corpus", further inquiries can be directed to the corresponding authors.
