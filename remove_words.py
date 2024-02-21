from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

import sys
import os
sys.path.append('./')
from utils.utils import clean_str, loadWord2Vec  


if len(sys.argv) != 2:
	sys.exit("Use: python remove_words.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'AMPs']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

print("-----------------1. 加载停用词！")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print(stop_words)

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
# vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])
# dataset = '20ng'


# 指定包含fold子文件夹的父文件夹路径
parent_dir = 'cross_validation_data'

# 获取所有fold子文件夹的名称
fold_folders = [folder for folder in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, folder))]

# 遍历每个fold子文件夹
for fold_folder in fold_folders:
    # 构建当前fold子文件夹内"corpus"文件夹的路径
    corpus_folder_path = os.path.join(parent_dir, fold_folder, 'data', 'corpus')

    # 检查该路径是否存在
    if os.path.exists(corpus_folder_path):
        # 打开corpus文件夹
        print(f"Processing corpus folder in {fold_folder}")

        # 加入您提供的文本输出代码块
        print("-----------------2. 加载语料库！")
        doc_content_list = []
        with open(os.path.join(corpus_folder_path, dataset + '.txt'), 'rb') as f:
            for line in f.readlines():
                doc_content_list.append(line.strip().decode('latin1'))  

        print("-----------------3. 统计词频！")
        word_freq = {}  # to remove rare words
        for doc_content in doc_content_list:
            temp = clean_str(doc_content)
            words = temp.split()
            for word in words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        print("-----------------4. 过滤低频词(出现次数小于5次的)")
        clean_docs = []
        for doc_content in doc_content_list:
            temp = clean_str(doc_content)
            words = temp.split()
            doc_words = []
            for word in words:
                # word not in stop_words and word_freq[word] >= 5
                if dataset == 'mr':
                    doc_words.append(word)
                elif word not in stop_words and word_freq[word] >= 5:
                    doc_words.append(word)

            doc_str = ' '.join(doc_words).strip()
            clean_docs.append(doc_str)

        clean_corpus_str = '\n'.join(clean_docs)

        print(F"-----------------5. 把高频词写入文件：data/corpus/{dataset}.clean.txt！")
        with open(os.path.join(corpus_folder_path, dataset + '.clean.txt'), 'w') as f:
            f.write(clean_corpus_str)

        # dataset = '20ng'
        min_len = 10000
        aver_len = 0
        max_len = 0

        print(F"-----------------6. 统计样本的最小、最大与平均长度！")
        with open(os.path.join(corpus_folder_path, dataset + '.clean.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                temp = line.split()
                aver_len = aver_len + len(temp)
                if len(temp) < min_len:
                    min_len = len(temp)
                if len(temp) > max_len:
                    max_len = len(temp)

        aver_len = 1.0 * aver_len / len(lines)
        print('Min_len : ' + str(min_len))
        print('Max_len : ' + str(max_len))
        print('Average_len : ' + str(aver_len))

        print(F"-----------------. 最终输出数据保存在文件：data/corpus/{dataset}.clean.txt中！")

    else:
        print(f"Corpus folder not found in {fold_folder}")

