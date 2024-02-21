"""
    处理数据集，并生成图卷积神经网络需要的训练与测试数据。
"""
import os

current_dir = os.path.dirname(__file__)

def word_frequency_and_vocab():
    data_file = "amps.data"                  # 分词后的正负特征文件
    targets_file = "amps.target"             # 使用索引对应特征的标签文件
    suffle_index_file = "shuffle.index"      # 打乱后的数据集（使用索引）

    # 数据集文件路径
    data_path = os.path.join(current_dir, "../data")  
    data_path_file    = os.path.join(data_path, data_file)
    targets_path_file = os.path.join(data_path, targets_file)
    shuffle_path_file = os.path.join(data_path, suffle_index_file)

    # 读取特征
    data = []
    targets = []
    shuffles = []
    with open(data_path_file, "r") as fd:
        lines = fd.readlines()
        for line in lines:
            data.append(line.strip())

    # 读取标签
    with open(targets_path_file, "r") as fd:
        lines = fd.readlines()
        for line in lines:
            targets.append(line.strip())

    # 读取打乱的数据集索引
    with open(shuffle_path_file, "r") as fd:
        lines = fd.readlines()
        for line in lines:
            shuffles.append(int(line.strip()))
    
    # 根据打乱索引得到打乱后的数据
    shuffle_data = []
    for shuffle_id in  shuffles:
        shuffle_data.append(data[shuffle_id])

    # 开始统计
    word_freq = {}      # 使用字典统计词频
    word_set = set()    # 使用集合作为词汇表
    for line in  shuffle_data:
        words = line.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word]  = 1
    # 词汇表
    vocab = list(word_set)     # 把词汇表转换为list
    # 词汇表长度
    vocab_size = len(vocab)    # 词汇总数
