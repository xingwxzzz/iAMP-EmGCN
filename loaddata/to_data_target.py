"""
把数据集转换为两个文件：
    amps.data    : 把正负样本的特征按照编号存放在amps.data  文件
    amps.target  : 把正负样本的标签按照编号存放在amps.target文件
转换为两个文件的时候，顺便分词：
    分词的规则：先尝试2肽，即长度为2的字符串为1个词
------------------------------------------------------------------
cut_words函数：
    对肽基因切词
read_data_from_file
    从数据集文件读取数据
save_and_shuffle_data
    保存特征与标签数据为两个文件，并同时打乱数据。
"""
import os
import random
from sklearn.model_selection import StratifiedKFold

current_dir = os.path.dirname(__file__)


def cut_words(line, cut_num=3):
    """
        把字符串，按照cut_num分词
    """
    new_line = ""
    for i in range(1, len(line) + 1):
        new_line += line[i-1]
        if i % 3 == 0:      #如果想切成3，这里改成3，
            new_line += " "
    return new_line

def read_data_from_file(data_file="AMPs.fasta", target_type=1):
    """
        data_file   : 数据集文件名
        target_type : 数据集文件是正样本1，还是负样本0
    """
    # 数据数据文件路径，数据文件必须放在data\corpus
    data_path = os.path.join(current_dir, "../data/corpus")        # 确定路径
    data_path_file = os.path.join(data_path, data_file)            # 数据文件路径
    # print(os.path.exists(data_path_file))

    # 读取文件内容，并解析
    data    = []    # 特征
    targets = []    # 标签
    # 打开文件
    with open(data_path_file, "r") as fd:
        # 读取文件                          
        lines = fd.readlines()
        # 处理数据
        for line in lines:                                         
            if line.startswith(">"):                               # 不处理 > 开头的行，这是注释说明行
                continue
            else:
                # 添加特征到列表（删除了\n，\t，空格等, 并切词）
                data.append(cut_words(line.strip(), cut_num=3).strip())
                # 添加标签到列表     
                targets.append(target_type)                                  
    return data, targets

def read_data_from_test_file(data_file="AMPs.fasta", target_type=1):
    """
        data_file   : 数据集文件名
        target_type : 数据集文件是正样本1，还是负样本0
    """
    # 数据数据文件路径，数据文件必须放在data\corpus
    data_path = os.path.join(current_dir, "../test_data/corpus")        # 确定路径
    data_path_file = os.path.join(data_path, data_file)            # 数据文件路径
    # print(os.path.exists(data_path_file))

    # 读取文件内容，并解析
    data    = []    # 特征
    targets = []    # 标签
    # 打开文件
    with open(data_path_file, "r") as fd:
        # 读取文件
        lines = fd.readlines()
        # 处理数据
        for line in lines:
            if line.startswith(">"):                               # 不处理 > 开头的行，这是注释说明行
                continue
            else:
                # 添加特征到列表（删除了\n，\t，空格等, 并切词）
                data.append(cut_words(line.strip(), cut_num=3).strip())
                # 添加标签到列表
                targets.append(target_type)
    return data, targets
def cross_validation_data(data, targets, num_folds=10, save_dir='cross_validation_data'):

    skf = StratifiedKFold(n_splits=num_folds)


    # 创建保存数据的目录
    os.makedirs(save_dir, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data, targets)):
        train_data = [data[i] for i in train_idx]
        train_targets = [targets[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        test_targets = [targets[i] for i in test_idx]

        # 保存数据到文件
        fold_dir = os.path.join(save_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        train_data_file = os.path.join(fold_dir, 'train_data.txt')
        train_targets_file = os.path.join(fold_dir, 'train_targets.txt')
        test_data_file = os.path.join(fold_dir, 'test_data.txt')
        test_targets_file = os.path.join(fold_dir, 'test_targets.txt')

        with open(train_data_file, 'w') as fd:
            for item in train_data:
                fd.write("%s\n" % item)

        with open(train_targets_file, 'w') as fd:
            for idx, item in enumerate(train_targets):
                fd.write(f"{idx}\t{'train'}\t{item}\n")

        with open(test_data_file, 'w') as fd:
            for item in test_data:
                fd.write("%s\n" % item)

        with open(test_targets_file, 'w') as fd:
            for idx, item in enumerate(test_targets):
                if len(test_targets) ==1811:
                    idx=idx+16303
                else:
                    idx = idx + 16302
                fd.write(f"{idx}\t{'test'}\t{item}\n")

        # 创建 "data" 子目录
        data_sub_dir = os.path.join(fold_dir, 'data')
        os.makedirs(data_sub_dir, exist_ok=True)

        # 创建 "corpus" 子目录
        corpus_sub_dir = os.path.join(data_sub_dir, 'corpus')
        os.makedirs(corpus_sub_dir, exist_ok=True)

        # 合并 train_data_file 和 test_data_file 到 "AMPs.txt" 位于 "corpus" 子目录下
        with open(os.path.join(corpus_sub_dir, 'AMPs.txt'), 'w') as fd:
            with open(train_data_file, 'r') as train_fd:
                fd.write(train_fd.read())
            with open(test_data_file, 'r') as test_fd:
                fd.write(test_fd.read())

        # 合并 train_targets_file 和 test_targets_file 到 "AMPs.txt" 位于 "data" 子目录下
        with open(os.path.join(data_sub_dir, 'AMPs.txt'), 'w') as fd:
            with open(train_targets_file, 'r') as train_fd:
                fd.write(train_fd.read())
            with open(test_targets_file, 'r') as test_fd:
                fd.write(test_fd.read())





def shuffle_and_save_data(data, targets):
    # 切分训练集与测试集，生成gcn实现需要的语料库
    new_data = []
    new_target = []

    for idx, target in enumerate(targets):
        # 数据特征
        new_data.append(data[idx])
        # 测试标签
        new_target.append(F"{idx}\t{'test'}\t{target}")

    new_data_str = "\n".join(new_data)
    new_target_str = "\n".join(new_target)

    data_file = "AMPs.txt"
    corpus_path = os.path.join(current_dir, "../test_data/corpus")
    target_path = os.path.join(current_dir, "../test_data")

    corpus_path_file = os.path.join(corpus_path, data_file)
    target_path_file = os.path.join(target_path, data_file)

    with open(corpus_path_file, "w") as fd:
        fd.write(new_data_str)

    with open(target_path_file, "w") as fd:
        fd.write(new_target_str)
