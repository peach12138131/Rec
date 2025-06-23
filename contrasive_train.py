import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras import layers, Model

import pickle
import pandas as pd
import numpy as np
from blocks import ResidualBlock, FeatureAttention, CrossLayer
from model import SupplierTower,CityTower,DualTowerModel,create_dual_tower_model
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import Adam
import time
import glob
try:
    from tensorflow.keras.saving import register_keras_serializable
except:
    # 对于较旧版本的TensorFlow
    from tensorflow.keras.utils import register_keras_serializable


# 修复版本的置信度监控回调类
class ConfidenceMonitor(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, pos_threshold=0.6, neg_threshold=0.4):
        super(ConfidenceMonitor, self).__init__()
        self.val_dataset = val_dataset  # 直接传入验证数据集
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # 收集验证数据集的预测和真实标签
        all_y_true = []
        all_y_pred = []
        
        # 遍历验证集进行预测
        for x, y in self.val_dataset:
            y_pred = self.model.predict(x, verbose=0)
            all_y_true.append(y.numpy())
            all_y_pred.append(y_pred)
        
        # 合并所有批次的结果
        if all_y_true and all_y_pred:
            val_y = np.concatenate(all_y_true, axis=0)
            y_pred = np.concatenate(all_y_pred, axis=0)
            
            # 计算正确的高置信度预测
            # 正类：真实为1且预测>=0.6
            high_conf_pos = np.logical_and(val_y == 1, y_pred >= self.pos_threshold)
            # 负类：真实为0且预测<0.4
            high_conf_neg = np.logical_and(val_y == 0, y_pred < self.neg_threshold)
            
            # 总的高置信度正确预测
            high_conf_correct = np.logical_or(high_conf_pos, high_conf_neg)
            high_conf_acc = np.mean(high_conf_correct)
            
            # 有多少正类被预测为高置信度正类
            true_pos = val_y == 1
            if np.sum(true_pos) > 0:
                high_conf_pos_rate = np.sum(high_conf_pos) / np.sum(true_pos)
            else:
                high_conf_pos_rate = 0
                
            # 有多少负类被预测为高置信度负类
            true_neg = val_y == 0
            if np.sum(true_neg) > 0:
                high_conf_neg_rate = np.sum(high_conf_neg) / np.sum(true_neg)
            else:
                high_conf_neg_rate = 0
                
            # 有多少预测落入中间区域（不确定区域）
            uncertain_preds = np.logical_and(y_pred >= self.neg_threshold, y_pred < self.pos_threshold)
            uncertain_rate = np.mean(uncertain_preds)
            
            # 打印结果
            print(f"\n--- Epoch {epoch+1} Confidence Metrics ---")
            print(f"High Confidence Accuracy: {high_conf_acc:.4f}")
            print(f"High Confidence Positive Rate: {high_conf_pos_rate:.4f}")
            print(f"High Confidence Negative Rate: {high_conf_neg_rate:.4f}")
            print(f"Uncertain Prediction Rate: {uncertain_rate:.4f}")





def create_model(model_path='./model/contrastive_model_0126_1.keras'):
    # 1. 加载编码器以获取词汇表大小
    with open('./data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

     ##拿到各个特征的大小,准备模型构造
    icao_size = len(encoders['icao_encoder'].classes_)
    city_size = len(encoders['city_encoder'].classes_)
    country_size = len(encoders['country_encoder'].classes_)
    area_size = len(encoders['area_encoder'].classes_)
    supplier_size = len(encoders['supplier_encoder'].classes_)
    aircraft_model_size = len(encoders['aircraft_encoder'].classes_)

    print(f"icao_size: {icao_size}, city_size: {city_size}, country_size: {country_size}, area_size: {area_size}, supplier_size: {supplier_size}, aircraft_model_size: {aircraft_model_size}")


    embedding_dim_input = 512
    num_heads_input = 4
    depth_mlp_input = 1
    depth_attention_input = 4
    units_input = 128
    output_dim = 128

    #vocab_size + 2 为特殊token
    model = create_dual_tower_model(
        icao_size=icao_size+2,
        city_size=city_size+2,
        country_size=country_size+2,
        area_size=area_size+2,
        supplier_size=supplier_size+2,
        aircraft_model_size=aircraft_model_size+2,
        embedding_dim=embedding_dim_input,
        num_heads=num_heads_input,
        depth_mlp=depth_mlp_input,
        depth_attention=depth_attention_input,
        units=units_input,
        output_dim=output_dim
    )

    model.summary()
    model.save(model_path)
    print(f"模型已保存为 {model_path}")

    



def create_augmented_sample(inputs, dropout_rate=0.2, swap_rate=0.2):
    """优化后的增强样本创建函数"""
    augmented = {}
    batch_size = tf.shape(inputs['supplier_id'])[0]
    
    for key, value in inputs.items():
      
        if key == 'supplier_id':
            augmented[key] = value
            continue
            
        # 1. 特征值交换 (移到最前面，因为它保持了值的有效性)
        random_indices = tf.random.shuffle(tf.range(batch_size))
        swap_mask = tf.random.uniform(shape=tf.shape(value)) < swap_rate
        swapped = tf.where(swap_mask, tf.gather(value, random_indices), value)
        
        # 2. 特征替换 (紧跟交换，同样保持值有效性)
        random_row_indices = tf.random.shuffle(tf.range(batch_size))
        replace_mask = tf.random.uniform(shape=tf.shape(value)) < 0.2
        replaced = tf.where(replace_mask, 
                          tf.gather(value, random_row_indices), 
                          swapped)
        
        # 3. 特征重复 (在值还有效时进行)
        shift_values = tf.roll(replaced, shift=1, axis=0)
        repeat_mask = tf.random.uniform(shape=tf.shape(value)) < 0.2
        repeated = tf.where(repeat_mask, shift_values, replaced)
        
        # 4. 特征dropout (在主要变换后进行)
        mask = tf.random.uniform(shape=tf.shape(value)) > dropout_rate
        dropped = tf.where(mask, repeated, 0)
        
        # 5. 特征遮蔽 (最后添加特殊token)
        mask_token = tf.constant(-1, dtype=value.dtype)
        mask_mask = tf.random.uniform(shape=tf.shape(value)) < 0.2
        masked = tf.where(mask_mask, mask_token, dropped)
        
        augmented[key] = masked
        
    return augmented

def info_nce_loss(original_embeddings, augmented_embeddings, temperature=0.1):
    """计算InfoNCE损失"""
    batch_size = tf.shape(original_embeddings)[0]
    
    # 计算相似度矩阵
    similarity_matrix = tf.matmul(
        tf.nn.l2_normalize(original_embeddings, axis=1),
        tf.nn.l2_normalize(augmented_embeddings, axis=1),
        transpose_b=True
    )
    
    # 正样本对的mask
    positive_mask = tf.eye(batch_size)
    
    # 调整温度系数
    similarity_matrix = similarity_matrix / temperature
    
    # 计算交叉熵损失
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=positive_mask,
        logits=similarity_matrix
    )
    
    return tf.reduce_mean(loss)  ##均值损失
    # return tf.reduce_sum(loss)  ##求和损失


def calculate_similarity_metrics(original_embeddings, augmented_embeddings):
    """
    计算相似度相关的监控指标，动态处理不同batch大小
    
    返回:
    - positive_similarities: 原始样本与其增广版本的相似度
    - negative_similarities: 原始样本与batch中其他样本的相似度
    - similarity_ratio: 正样本相似度与负样本相似度的比值
    """
    # 计算正样本对的相似度 (原始与增广版本)
    positive_similarities = tf.reduce_sum(
        tf.multiply(original_embeddings, augmented_embeddings), axis=1
    )
    
    # 计算负样本对的相似度 (与batch中其他样本)
    # 创建所有可能的样本对
    current_batch_size = tf.shape(original_embeddings)[0]
    embeddings_2d = tf.reshape(original_embeddings, (current_batch_size, 1, -1))
    augmented_2d = tf.reshape(augmented_embeddings, (1, current_batch_size, -1))
    
    # 计算余弦相似度矩阵
    similarities = tf.reduce_sum(
        tf.multiply(embeddings_2d, augmented_2d), axis=2
    )
    
    # 移除对角线(正样本对)
    mask = tf.ones_like(similarities) - tf.eye(current_batch_size)
    negative_similarities = tf.boolean_mask(similarities, mask > 0)
    
    # 计算正负样本相似度比值
    avg_positive_sim = tf.reduce_mean(positive_similarities)
    avg_negative_sim = tf.reduce_mean(negative_similarities)
    similarity_ratio = avg_positive_sim / (avg_negative_sim + 1e-8)
    
    return {
        'avg_positive_similarity': avg_positive_sim,
        'avg_negative_similarity': avg_negative_sim,
        'similarity_ratio': similarity_ratio
    }



def contrastive_train(batch_size,epochs,model_path):
    """
    对比学习训练，为了消除航司向量各项异性，引入对比学习
    学习方案：对supplier_tower,对某一个supplier数据 a，对其输入特征进行变化。那么变化后的a1输出向量，应该和原始a向量相似，和其他b的向量不相似

    变化方案：1.除了supplier id ，每一个其他特征有概率50%被dropout
    2.为每个特征，50%概率加入噪声
    
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split

    ##用dataset 管理数据集，避免显存溢出
    def create_dataset(inputs, batch_size, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(inputs['supplier_id']))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    tf.keras.backend.clear_session()  ##起手清一下显存
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 启用显存增长
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # 设置初始限制为28GB
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=28672)]
            )
            print("当前使用GPU为：",gpus[0])
        except RuntimeError as e:
            print(e)

      # 1. 加载编码器以获取词汇表大小
    with open('./data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)


     # 加载预处理后的数据,再随机打乱
    print("加载预处理后的数据")
    # 获取所有以encoded.csv结尾的文件
    all_files = glob.glob("./data/*encoded.csv")
    
    # 读取并合并所有文件
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    # 合并所有数据框
    encoded_df = pd.concat(df_list, ignore_index=True)
    print(f"成功加载并合并所有数据")
  
    suppliers_df = encoded_df.drop_duplicates(subset=['supplier_id'], keep='first').sample(frac=1).reset_index(drop=True)
    print(suppliers_df)

    del encoded_df #数据准备完成


     #准备输入数据,只训练航司supplier
    inputs={
        
        #航司塔输入
        'supplier_id': suppliers_df['supplier_id'].values,
        'supplier_country_name': suppliers_df['supplier_country_name'].values,
        'supplier_aircraft_scale_type':suppliers_df['supplier_aircraft_scale_type'].values,
    }


    for i in range(1, 4):
        inputs[f'main_area_{i}'] = suppliers_df[f'main_area_{i}'].values

    # 添加机型特征 (15个机型)
    for i in range(1, 16):
        inputs[f'aircraft_model_{i}'] = suppliers_df[f'aircraft_model_{i}'].values

    for i in range(1, 7):
        inputs[f'base_{i}'] = suppliers_df[f'base_{i}'].values

    # for i in range(1, 7):
    #     inputs[f'base_city_{i}'] = suppliers_df[f'base_city_{i}'].values

    # 数据集划分
    val_size = 0.2
    train_size = int(len(suppliers_df) * (1-val_size))


    # 创建训练和验证数据集
    train_inputs = {k: v[:train_size] for k,v in inputs.items()}
    val_inputs = {k: v[train_size:] for k,v in inputs.items()}

    train_dataset = create_dataset(train_inputs, batch_size)
    val_dataset = create_dataset(val_inputs, batch_size)





    original_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'DualTowerModel': DualTowerModel,
            'supplierTower': SupplierTower,
            'CityTower': CityTower,
            'ResidualBlock': ResidualBlock,
            'FeatureAttention': FeatureAttention,
            'CrossLayer': CrossLayer
        }
    )
 
    original_model.summary()

  
    supplier_model=original_model.supplier_tower  ##对比学习只训练航司塔，剩余的塔先不管

    # 优化器设置
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    initial_learning_rate = 1e-4
    steps_per_epoch = len(train_dataset)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=steps_per_epoch,
        t_mul=1.0,  # 每次重启后周期的倍数
        m_mul=0.9,  # 每次重启后学习率最大值的倍数
        alpha=0.000001  # 最小学习率比例
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


     # 创建记录指标的列表
    train_metrics_history = {
        'loss': [], 
        'avg_positive_similarity': [],
        'avg_negative_similarity': [],
        'similarity_ratio': []
    }
    val_metrics_history = {
        'loss': [],
        'avg_positive_similarity': [],
        'avg_negative_similarity': [],
        'similarity_ratio': []
    }

    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        # 训练指标
        epoch_metrics = {
            'loss': [],
            'avg_positive_similarity': [],
            'avg_negative_similarity': [],
            'similarity_ratio': []
        }
        
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                # 原始embedding
                original_embeddings = supplier_model(batch)
                
                # 增强样本的embedding
                augmented_batch = create_augmented_sample(batch)
                augmented_embeddings = supplier_model(augmented_batch)
                
                # 计算对比学习损失
                loss = info_nce_loss(original_embeddings, augmented_embeddings)
                
                # 计算相似度指标
                similarity_metrics = calculate_similarity_metrics(
                    original_embeddings, 
                    augmented_embeddings
                )
            
            # 梯度更新
            gradients = tape.gradient(loss, supplier_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, supplier_model.trainable_variables))
            
            # 记录指标
            epoch_metrics['loss'].append(loss)
            for metric_name, value in similarity_metrics.items():
                epoch_metrics[metric_name].append(value)
            
            # 打印batch进度
            if num_batches % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}")
                print(f"Loss: {loss:.4f}")
                for metric_name, value in similarity_metrics.items():
                    print(f"{metric_name}: {value:.4f}")
        
        # 计算epoch平均指标
        avg_metrics = {
            name: tf.reduce_mean(values).numpy()
            for name, values in epoch_metrics.items()
        }
        
        # 记录训练指标历史
        for metric_name, value in avg_metrics.items():
            train_metrics_history[metric_name].append(value)
        
        # 验证阶段
        val_epoch_metrics = {
            'loss': [],
            'avg_positive_similarity': [],
            'avg_negative_similarity': [],
            'similarity_ratio': []
        }
        
        for val_batch in val_dataset:
            # 计算验证指标
            val_embeddings = supplier_model(val_batch)
            val_augmented = create_augmented_sample(val_batch)
            val_aug_embeddings = supplier_model(val_augmented)
            
            val_loss = info_nce_loss(val_embeddings, val_aug_embeddings)
            val_similarity_metrics = calculate_similarity_metrics(
                val_embeddings,
                val_aug_embeddings
            )
            
            val_epoch_metrics['loss'].append(val_loss)
            for metric_name, value in val_similarity_metrics.items():
                val_epoch_metrics[metric_name].append(value)
        
        # 计算验证集平均指标
        avg_val_metrics = {
            name: tf.reduce_mean(values).numpy()
            for name, values in val_epoch_metrics.items()
        }
        
        # 记录验证指标历史
        for metric_name, value in avg_val_metrics.items():
            val_metrics_history[metric_name].append(value)
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("\nTraining Metrics:")
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print("\nValidation Metrics:")
        for metric_name, value in avg_val_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print("\n" + "="*50 + "\n")
    
    # 保存模型
    original_model.supplier_tower.set_weights(supplier_model.get_weights())
    original_model.save(model_path)
    print(f"模型已保存为 {model_path}")
    
    # 返回训练历史，便于后续绘图分析
    return train_metrics_history, val_metrics_history



def plot_training_history(train_history, val_history):
    """
    avg_positive_similarity: 计算原始样本与其增广版本之间的平均相似度。这个值越高，说明模型越能保持增广前后的一致性。
avg_negative_similarity: 计算原始样本与batch中其他样本之间的平均相似度。这个值应该较低，表示不同航司之间的区分度。
similarity_ratio: 正样本相似度与负样本相似度的比值。这个比值越大，说明模型越能够区分相似和不相似的样本对。
    """
    import matplotlib.pyplot as plt
    
    metrics = ['loss', 'avg_positive_similarity', 'avg_negative_similarity', 'similarity_ratio']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.plot(train_history[metric], label='Train')
        ax.plot(val_history[metric], label='Validation')
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('./image_contrastive_train.png', format='png')  # 保存为 PNG 








def pos_contrastive_train(batch_size,epochs,model_path,model,fintune_pattern,fintune):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split

    tf.keras.backend.clear_session()  ##起手清一下显存
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 启用显存增长
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # 设置初始限制为28GB
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=28672)]
            )
            print("当前使用GPU为：",gpus[0])
        except RuntimeError as e:
            print(e)

      # 1. 加载编码器以获取词汇表大小
    with open('./data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    print("加载预处理后的数据")
    all_files = glob.glob(f"./data/*encoded.csv")
    
    # 读取并合并所有文件
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)

       
    encoded_df = pd.concat(df_list, ignore_index=True)


 

    def prepare_data(data_dir="./data/", fintune_pattern=fintune_pattern, encoded_df=encoded_df,sample_frac=1, random_state=42,fintune=fintune):
        """
        准备微调数据集，包括：
        1. 使用通配符选择含有特定字样的文件作为微调数据
        2. 从微调数据中随机抽取一定比例的正样本(label=1)
        3. 从非微调数据中随机抽取相同数量的正样本
        4. 组合微调和非微调数据的特定列，形成组合负样本
        5. 将新创建的负样本添加到原始微调数据集中
        
        参数:
        data_dir (str): 数据文件所在目录
        asia_pattern (str): 用于选择微调文件的通配符模式
        sample_frac (float): 从微调数据的正样本中抽取的比例
        random_state (int): 随机数种子
        
        返回:
        pandas.DataFrame: 处理后的微调数据集
        
        """

        if fintune==False:

            return encoded_df
       

        elif fintune == True:
        
            # 使用通配符选择含有特定字样的文件进行微调
            fine_tune_files = glob.glob(f"{data_dir}/{fintune_pattern}")
            fine_tune_dfs = []
            for file in fine_tune_files:
                df = pd.read_csv(file)
                fine_tune_dfs.append(df)
            
            # 合并所有微调数据框
            if fine_tune_dfs:
                fine_tune_df = pd.concat(fine_tune_dfs, ignore_index=True)
                fine_tune_df = fine_tune_df.sample(frac=1.0, random_state=42).reset_index(drop=True)  ##打乱数据
                # print(f"微调数据加载完成，共{len(fine_tune_df)}条记录")
                # return fine_tune_df
            else:
                print("未找到匹配的微调数据文件")
                return None
            
            # 获取非微调数据文件列表
            non_fine_tune_files = [file for file in all_files if file not in fine_tune_files]
            non_fine_tune_dfs = []
            for file in non_fine_tune_files:
                df = pd.read_csv(file)
                non_fine_tune_dfs.append(df)
            
            # 合并所有非微调数据框
            if non_fine_tune_dfs:
                non_fine_tune_df = pd.concat(non_fine_tune_dfs, ignore_index=True)
                print(f"非微调数据加载完成，共{len(non_fine_tune_df)}条记录")
            else:
                print("未找到非微调数据文件")
                return fine_tune_df  # 如果没有非微调数据，直接返回微调数据
            
            # 从微调数据中随机抽取一定比例的label=1的数据
            fine_tune_positive = fine_tune_df[fine_tune_df['label'] == 1]
            sampled_fine_tune = fine_tune_positive.sample(frac=sample_frac, random_state=random_state)
            print(f"从微调数据中抽取了{len(sampled_fine_tune)}条正样本")
            
            # 从非微调数据中随机抽取相同数量的label=1的数据
            non_fine_tune_positive = non_fine_tune_df[non_fine_tune_df['label'] == 1]
            
            # 确保非微调正样本数量足够
            if len(non_fine_tune_positive) < len(sampled_fine_tune):
                print(f"警告：非微调数据中的正样本数量({len(non_fine_tune_positive)})少于所需数量({len(sampled_fine_tune)})")
                sampled_non_fine_tune = non_fine_tune_positive
            else:
                sampled_non_fine_tune = non_fine_tune_positive.sample(n=len(sampled_fine_tune), random_state=random_state)
            
            print(f"从非微调数据中抽取了{len(sampled_non_fine_tune)}条正样本")
            
            # 创建组合负样本
            # 从微调数据中提取特定列
            cols_from_fine_tune = ['dep_icao', 'arr_icao', 'dep_country_name', 'arr_country_name', 
                                   'dep_area_name', 'arr_area_name']
            
            # 确保所有需要的列都存在
            for col in cols_from_fine_tune:
                if col not in sampled_fine_tune.columns:
                    print(f"错误：微调数据中缺少列 '{col}'")
                    return fine_tune_df
            
            fine_tune_part = sampled_fine_tune[cols_from_fine_tune].reset_index(drop=True)
            
            # 从非微调数据中提取其他列
            other_cols = [col for col in sampled_non_fine_tune.columns if col not in cols_from_fine_tune]
            non_fine_tune_part = sampled_non_fine_tune[other_cols].reset_index(drop=True)
            
            # 拼接创建组合负样本
            combined_negative = pd.concat([fine_tune_part, non_fine_tune_part], axis=1)
            combined_negative['label'] = 0  # 设置为负样本
            
            print(f"生成了{len(combined_negative)}条组合负样本")
            
            # 合并原微调数据和新创建的负样本
            final_fine_tune_df = pd.concat([fine_tune_df, combined_negative], ignore_index=True)
            print(f"最终微调数据集大小: {len(final_fine_tune_df)}条记录")
            
            
            
            # 打印最终数据集的标签分布
            print("最终数据集标签分布:")
            print(final_fine_tune_df['label'].value_counts())
            
            return final_fine_tune_df

    
   
    
       # 创建要处理的文件名列表
    file_patterns = [
        "from_Africa_Africa_encoded.csv",
        "from_Central America_Central America_encoded.csv",
        "from_Central America_North America_encoded.csv",
        "from_East Asia_East Asia_encoded.csv",
        "from_Europe_Europe_encoded.csv",
        "from_Europe_West Asia_encoded.csv",
        "from_North America_Central America_encoded.csv",
        "from_North America_North America_encoded.csv",
        "from_Oceania_Oceania_encoded.csv",
        "from_South America_South America_encoded.csv",
        "from_Southeast Asia_Southeast Asia_encoded.csv",
        "from_West Asia_Europe_encoded.csv",
        "from_West Asia_West Asia_encoded.csv",
        "others_encoded.csv"
    ]
    
    # 创建一个字典来存储特定文件的采样率（如果有不同的采样率）
    sample_rates = {
        "from_North America_North America_encoded.csv": 0.8,
        "from_North America_Central America_encoded.csv": 0.8,
        "from_Central America_North America_encoded.csv": 0.8,
        "from_South America_South America_encoded.csv": 0.8
        # 添加其他需要特殊采样率的文件
    }
    
    # 存储所有处理后的数据框
    dfs = []
    
    # 循环处理每个文件
    for file_pattern in file_patterns:
        # 获取对应的采样率，如果没有特别指定，则使用默认值1.0
        sample_frac = sample_rates.get(file_pattern, 1.0)
        
        # 处理数据
        df = prepare_data(
            data_dir="./data/", 
            fintune_pattern=file_pattern, 
            encoded_df=encoded_df,
            sample_frac=sample_frac, 
            random_state=42,
            fintune=fintune
        )
        
        # 将处理后的数据添加到列表中
        dfs.append(df)
        
        # 可选：打印进度
        print(f"处理完成: {file_pattern}, 采样率: {sample_frac}, 行数: {len(df)}")
    
    # 合并所有数据框
    encoded_df = pd.concat(dfs, ignore_index=True)
    
    # 打乱数据
    encoded_df = encoded_df.sample(frac=1.0, random_state=42).reset_index(drop=True)


    
    




    #准备输入数据
    inputs={
        #城市塔
        'dep_icao': encoded_df['dep_icao'].values,
        'arr_icao': encoded_df['arr_icao'].values,
        'dep_city_name': encoded_df['dep_city_name'].values,
        'arr_city_name': encoded_df['arr_city_name'].values,
        'dep_country_name': encoded_df['dep_country_name'].values,
        'arr_country_name': encoded_df['arr_country_name'].values,
        'dep_area_name': encoded_df['dep_area_name'].values,
        'arr_area_name': encoded_df['arr_area_name'].values,


        #航司塔输入
        'supplier_id': encoded_df['supplier_id'].values,
        'supplier_country_name': encoded_df['supplier_country_name'].values,
        'supplier_aircraft_scale_type':encoded_df['supplier_aircraft_scale_type'].values,
    }


    for i in range(1, 4):
        inputs[f'main_area_{i}'] = encoded_df[f'main_area_{i}'].values

    # 添加机型特征 (15个机型)
    for i in range(1, 16):
        inputs[f'aircraft_model_{i}'] = encoded_df[f'aircraft_model_{i}'].values

    for i in range(1, 7):
        inputs[f'base_{i}'] = encoded_df[f'base_{i}'].values

    # for i in range(1, 7):
    #     inputs[f'base_city_{i}'] = encoded_df[f'base_city_{i}'].values

    y = encoded_df['label'].values
    print("标签")
    print(len(y))

    # 计算0和1的数量
    num_zeros = (y == 0).sum()
    num_ones = (y == 1).sum()
    
    print(f"标签0的数量: {num_zeros}")
    print(f"标签1的数量: {num_ones}")
    print(f"标签分布比例 (0:1): {num_zeros}:{num_ones}")


     #  划分训练集、验证集和测试集
    temp_indices, test_indices = train_test_split(
        range(len(y)), test_size=0.2, random_state=42
    )


    train_indices, val_indices = train_test_split(
        temp_indices, test_size=0.25, random_state=42
    )

     # 准备训练集、验证集和测试集的输入和标签
    train_inputs = {k: v[train_indices] for k, v in inputs.items()}
    val_inputs = {k: v[val_indices] for k, v in inputs.items()}
    test_inputs = {k: v[test_indices] for k, v in inputs.items()}

    train_y = y[train_indices]
    val_y = y[val_indices]
    test_y = y[test_indices]


    ##用dataset 管理数据集，避免显存溢出
    def create_dataset(inputs, labels, batch_size, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(labels))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_dataset(train_inputs, train_y, batch_size, shuffle=False)
    val_dataset = create_dataset(val_inputs, val_y, batch_size, shuffle=False)
    test_dataset = create_dataset(test_inputs, test_y, batch_size, shuffle=False)

    del encoded_df  # 数据准备完成，删除df节省内存
    



    





    # 3. 设置学习率调度
    if fintune==False:
        initial_learning_rate = 1e-6
    elif fintune==True:
        initial_learning_rate = 1e-6
    t_mul = 1.0                  # 每次重启后周期的倍数
    m_mul = 0.9                  # 每次重启后学习率最大值的倍数
    alpha = 0.01                  # 最小学习率比例

    batch_size=batch_size
    epochs=epochs
    
    
    # 计算每个epoch的步数
    steps_per_epoch = len(train_dataset)  # Dataset自动计算批次数

    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=steps_per_epoch,  
        t_mul=t_mul,
        m_mul=m_mul,
        alpha=alpha
    )
    
    # 使用带调度的优化器
    optimizer = Adam(learning_rate=lr_schedule,clipnorm=1.0)


    class TrainingMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            print(f"Training Loss: {logs['loss']:.4f}")
            print(f"Validation Loss: {logs['val_loss']:.4f}")
            print(f"Training Accuracy: {logs['binary_accuracy']:.4f}")
            print(f"Validation Accuracy: {logs['val_binary_accuracy']:.4f}")

    class ReinforcementMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            print(f"Training Loss: {logs['loss']:.4f}")
            print(f"Training Accuracy: {logs['binary_accuracy']:.4f}")
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )


  
   
    

   
    class ReinforcementRuleCallback(tf.keras.callbacks.Callback):
        def __init__(self, train_dataset, train_inputs, train_y):
            super(ReinforcementRuleCallback, self).__init__()
            self.train_dataset = train_dataset
            self.train_inputs = train_inputs
            self.train_y = train_y
            self.modified_indices = []
        
        def on_epoch_end(self, epoch, logs=None):
        
            print("\n应用强化规则...")
            
            # 获取训练数据的预测结果
            predictions = self.model.predict(self.train_inputs)
            
            # 追踪修改的样本数量
            modified_count = 0
            
            # 创建原始数据的副本进行修改
            modified_train_y = np.copy(self.train_y)
            
            # 收集特殊样本索引
            special_indices = []
            
            for i in range(len(self.train_y)):
                # 检查样本是否被预测为正样本 (概率 > 0.5)
                if predictions[i] > 0.5:
                    # 获取当前样本的dep_icao和arr_icao
                    dep_icao_area = self.train_inputs['dep_area_name'][i]
                    arr_icao_area = self.train_inputs['arr_area_name'][i]
                    
                    # 获取当前样本的主要区域
                    main_area_1 = self.train_inputs['main_area_1'][i]
                    main_area_2 = self.train_inputs['main_area_2'][i]
                    main_area_3 = self.train_inputs['main_area_3'][i]

                    
                    
                    is_dep_in_main_areas = (dep_icao_area in [main_area_1, main_area_2, main_area_3])
                    is_arr_in_main_areas = (arr_icao_area in [main_area_1, main_area_2, main_area_3])
                    
                    # 应用规则：如果出发地和目的地都不在主要区域内，且当前标签为0
                    if not (is_dep_in_main_areas or is_arr_in_main_areas) and modified_train_y[i] == 0:
                        # 将标签更改为0（实际上不需要改变，因为已经是0）
                        modified_train_y[i] = 0
                        modified_count += 1
                        self.modified_indices.append(i)
                        special_indices.append(i)
            
            if modified_count > 0:
                print(f"根据强化规则发现了{modified_count}个样本")
                
                # 收集标签为1但预测概率小于0.5的样本（模型误判的正样本）
                misclassified_positives = [i for i in range(len(self.train_y)) 
                                        if self.train_y[i] == 1 and predictions[i] < 0.5]
                
                print(f"发现{len(misclassified_positives)}个被误判的正样本")
                
                # 查找所有真实标签为1的样本
                positive_indices = [i for i in range(len(self.train_y)) if self.train_y[i] == 1]
                
                # 决定要使用的正样本
                if len(misclassified_positives) >= modified_count:
                    # 如果误判的正样本足够，从中随机选择需要的数量
                    selected_positive_indices = np.random.choice(
                        misclassified_positives, 
                        size=modified_count, 
                        replace=False
                    )
                    print(f"从误判的正样本中选择了{len(selected_positive_indices)}个样本")
                else:
                    # 如果误判的正样本不足，先全部使用这些误判样本
                    selected_positive_indices = misclassified_positives
                    
                    # 计算还需要多少个正样本
                    remaining_needed = modified_count - len(misclassified_positives)
                    
                    # 从未被误判的正样本中选择剩余所需数量
                    correctly_classified_positives = [i for i in positive_indices if i not in misclassified_positives]
                    
                    if len(correctly_classified_positives) <= remaining_needed:
                        # 如果剩余的正样本不足，使用所有可用的正样本
                        additional_positives = correctly_classified_positives
                        print(f"使用所有{len(misclassified_positives)}个误判正样本和{len(additional_positives)}个随机正样本")
                    else:
                        # 随机选择剩余所需的正样本
                        additional_positives = np.random.choice(
                            correctly_classified_positives, 
                            size=remaining_needed, 
                            replace=False
                        )
                        print(f"使用所有{len(misclassified_positives)}个误判正样本和{len(additional_positives)}个随机正样本")
                    
                    # 合并两部分正样本
                    selected_positive_indices = np.concatenate([selected_positive_indices, additional_positives])
                
                # 合并特殊样本和正样本索引
                balanced_indices = np.concatenate([special_indices, selected_positive_indices])
                np.random.shuffle(balanced_indices)  # 打乱顺序
                
                # 创建平衡的训练数据集
                balanced_inputs = {k: v[balanced_indices] for k, v in self.train_inputs.items()}
                balanced_labels = self.train_y[balanced_indices]
                
                # 创建平衡的训练数据集
                balanced_dataset = create_dataset(
                    balanced_inputs, 
                    balanced_labels, 
                    batch_size, 
                    shuffle=True
                )
                
                # 使用平衡数据集进行强化训练
                print(f"使用平衡数据集({len(balanced_indices)}个样本)进行聚焦训练...")
                self.model.fit(
                    balanced_dataset,
                    epochs=1,
                    verbose=1,
                    callbacks=[ReinforcementMonitor()]
                )
                
                # 更新原始训练数据，用于未来的epoch
                self.train_y = modified_train_y
                
                # 使用更新后的标签重新创建主训练数据集
                self.train_dataset = create_dataset(
                    self.train_inputs, 
                    self.train_y, 
                    batch_size, 
                    shuffle=True
                )
                
                print(f"强化规则应用完成。到目前为止共发现了{len(set(self.modified_indices))}个样本")
            else:
                print("未发现符合条件的样本，跳过强化训练")
                
    reinforcement_callback = ReinforcementRuleCallback(
                train_dataset=train_dataset,
                train_inputs=train_inputs,
                train_y=train_y
            )

    class GradientMonitor(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if batch % 100 == 0:  # 每100批次检查一次
                # 获取梯度
                weights = self.model.trainable_weights
                with tf.GradientTape() as tape:
                    # 获取一个批次
                    for x_batch, y_batch in train_dataset.take(1):
                        y_pred = self.model(x_batch, training=True)
                        # 使用适当的损失函数，而不是model.loss
                        loss_fn = tf.keras.losses.get(self.model.loss) if isinstance(self.model.loss, str) else self.model.loss
                        loss = loss_fn(y_batch, y_pred)
                grads = tape.gradient(loss, weights)
                # 打印梯度统计信息
                for i, (w, g) in enumerate(zip(weights, grads)):
                    if g is not None:
                        g_norm = tf.norm(g).numpy()
                        if np.isnan(g_norm) or np.isinf(g_norm):
                            print(f"警告: 层 {i} 的梯度是 NaN 或 Inf!")
                        elif g_norm > 10.0:
                            print(f"警告: 层 {i} 的梯度范数过大: {g_norm}")
                    else:
                        print(f"警告: 层 {i} 的梯度为None")

    # 在训练时使用此回调
    gradient_monitor = GradientMonitor()
    

 

    callbacks = [
        TrainingMonitor(),
        # confidence_monitor,
        reinforcement_callback
        # gradient_monitor
    ]


 
    # 8. 训练模型
    history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    verbose=1,
    callbacks=callbacks
    )

    # 保存完整模型
    model.save(model_path)
    print(f"模型已保存为 {model_path}")


    #  # 9. 评估模型
    # # 在测试集上进行预测
    # test_predictions_prob = model.predict(test_dataset)
    # test_predictions_binary = (test_predictions_prob > 0.5).astype(int)

    # # 计算各种指标
    # accuracy = accuracy_score(test_y, test_predictions_binary)
    # precision = precision_score(test_y, test_predictions_binary)
    # recall = recall_score(test_y, test_predictions_binary)
    # f1 = f1_score(test_y, test_predictions_binary)

    # from sklearn.metrics import confusion_matrix, roc_auc_score

    # # 训练后的损失和精度
    # train_loss = history.history['loss']
    # train_accuracy = history.history['binary_accuracy']
    # val_loss = history.history['val_loss']
    # val_accuracy = history.history['val_binary_accuracy']


    # print("训练和验证集：=====================================================================")

    # # 打印所有训练过程中每个epoch的损失和精度
    # for epoch in range(len(train_loss)):
    #     print(f"Epoch {epoch + 1}: "
    #           f"Train Loss: {train_loss[epoch]:.4f}, Train Accuracy: {train_accuracy[epoch]:.4f}, "
    #           f"Val Loss: {val_loss[epoch]:.4f}, Val Accuracy: {val_accuracy[epoch]:.4f}")

    # print("测试集：===========================================================================")

    # # 计算混淆矩阵
    # cm = confusion_matrix(test_y, test_predictions_binary)
    # print("\n=== 混淆矩阵 ===")
    # print(cm)

    # # 计算AUC
    # auc = roc_auc_score(test_y, test_predictions_prob)
    # print(f"\nAUC: {auc:.4f}")

    # from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

    # # 绘制ROC曲线
    # fpr, tpr, thresholds = roc_curve(test_y, test_predictions_prob)
    # roc_auc = auc(fpr, tpr)
    # print("\n thresholds : \n")
    # print(thresholds)




    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 2, 1)
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
  

   

    # print("\n=== 模型评估结果 ===")
    # print(f"准确率 (Accuracy): {accuracy:.4f}")
    # print(f"精确率 (Precision): {precision:.4f}")
    # print(f"召回率 (Recall): {recall:.4f}")
    # print(f"F1分数: {f1:.4f}")


    # # 保存完整模型
    # model.save(model_path)
    # print(f"模型已保存为 {model_path}")

   
    # # 10. 绘制训练历史
    # # 损失曲线
    # plt.subplot(2, 2, 2)
    # plt.plot(history.history['loss'], label='train_loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.title('model_loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # # 准确率曲线
    # plt.subplot(2, 2, 3)
    # plt.plot(history.history['binary_accuracy'], label='train_accuracy')
    # plt.plot(history.history['val_binary_accuracy'], label='val_accuracy')
    # plt.title('model_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.tight_layout()
    # # plt.show()

    # plt.savefig('./image_post_contrastive_model_1.png', format='png')  # 保存为 PNG 






if __name__ == "__main__":
    model_path='./model/contrastive_model_0225.keras'
    # create_model(model_path)
    # train_history, val_history = contrastive_train(batch_size=512,epochs=20,model_path=model_path)
    # plot_training_history(train_history, val_history)



    ##加载对比学习模型
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'DualTowerModel': DualTowerModel,
            'supplierTower': SupplierTower,
            'CityTower': CityTower,
            'ResidualBlock': ResidualBlock,
            'FeatureAttention': FeatureAttention,
            'CrossLayer': CrossLayer
        }
    )
    print(model)
    model.summary()
    
    pos_contrastive_train(batch_size=512,epochs=10,model_path=model_path,model=model,fintune_pattern="*Asia*encoded.csv",fintune=True)
        

  


    


    








