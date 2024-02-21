from loaddata.to_data_target import read_data_from_file, cross_validation_data

if __name__ == "__main__":
    # 读取正样本
    data_pos, targets_pos = read_data_from_file(data_file="AMPs.fasta",    target_type=1) 
    # 读取负样本
    data_neg, targets_neg = read_data_from_file(data_file="nonAMPs.fasta", target_type=0) 
    
    # 合并正负样本
    data = data_pos + data_neg
    targets = targets_pos + targets_neg

    # 保存合并的特征与标签，并产生一个随机洗牌的数据集索引
    cross_validation_data(data, targets, num_folds=10, save_dir='cross_validation_data')
    #shuffle_and_save_data(data=data, targets=targets, split_rate=0.2)

