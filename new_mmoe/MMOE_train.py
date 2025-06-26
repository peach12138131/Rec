import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras import layers, Model

import pickle
import pandas as pd
import numpy as np
from blocks import ResidualBlock, FeatureAttention, CrossLayer
from model import MMOE,create_MMOE,get_input
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import Adam
import time
import glob
try:
    from tensorflow.keras.saving import register_keras_serializable
except:
    # 对于较旧版本的TensorFlow
    from tensorflow.keras.utils import register_keras_serializable

class TrainingMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(f"Training Loss: {logs['loss']:.4f}")
        print(f"Validation Loss: {logs['val_loss']:.4f}")
        print(f"CTR Training Accuracy: {logs['ctr_output_accuracy']:.4f}")
        print(f"CTR Validation Accuracy: {logs['val_ctr_output_accuracy']:.4f}")
        print(f"Aircraft Training Accuracy: {logs['aircraft_output_accuracy']:.4f}")
        print(f"Aircraft Validation Accuracy: {logs['val_aircraft_output_accuracy']:.4f}")

class ReinforcementMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(f"Reinforcement Training Loss: {logs['loss']:.4f}")
        print(f"CTR Training Accuracy: {logs['ctr_output_accuracy']:.4f}")




def prepare_data(data_dir="./data/", fintune_pattern="*Asia*encoded.csv", encoded_df=None,all_files=None,sample_frac=1, random_state=42,fintune=True):
        """
        准备微调数据集，包括：
        1. 使用通配符选择含有特定字样的文件作为微调数据
        2. 从微调数据中随机抽取一定比例的正样本(label=1)
        3. 从非微调数据中随机抽取相同数量的正样本
        4. 组合微调和非微调数据的特定列，形成组合负样本，
        5. 将新创建的负样本添加到原始微调数据集中
        
        参数:
        data_dir (str): 数据文件所在目录
        asia_pattern (str): 用于选择微调文件的通配符模式
        sample_frac (float): 从微调数据的正样本中抽取的比例
        random_state (int): 随机数种子
        all_files (list): 所有数据文件的列表
        encoded_df (pandas.DataFrame): 预处理后的数据集
        
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
               
            else:
                print("未找到匹配的微调数据文件")
                return None
            
            
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
                                   'dep_area_name', 'arr_area_name','flight_distance_bin']
            
            # 确保所有需要的列都存在
            for col in cols_from_fine_tune:
                if col not in sampled_fine_tune.columns:
                    print(f"错误：微调数据中缺少列 '{col}'")
                    return fine_tune_df
            
            fine_tune_part = sampled_fine_tune[cols_from_fine_tune].reset_index(drop=True)  #从区域里抽取
            
            # 从非微调数据中提取其他列
            other_cols = [col for col in sampled_non_fine_tune.columns if col not in cols_from_fine_tune]
            non_fine_tune_part = sampled_non_fine_tune[other_cols].reset_index(drop=True) #从其他区域抽取
            
            # 拼接创建组合负样本
            combined_negative = pd.concat([fine_tune_part, non_fine_tune_part], axis=1)
            combined_negative['label'] = 0  # 设置为负样本，aircraft_model保持原样
            
            print(f"生成了{len(combined_negative)}条组合负样本")
            
            # 合并原微调数据和新创建的负样本
            final_fine_tune_df = pd.concat([fine_tune_df, combined_negative], ignore_index=True)
            print(f"最终微调数据集大小: {len(final_fine_tune_df)}条记录")
            
            
            
        
            print("最终数据集标签分布:")
            print(final_fine_tune_df['label'].value_counts())
            
            return final_fine_tune_df




def MMOE_train_real(batch_size,epochs,model_path,model,fintune_pattern,fintune):
    """"
    训练思路1：区域性训练，随机负采样，从其他区域随机采样负样本
    训练思路2：强化规则训练，强化规则：如果出发地和目的地都不在主要区域内，且当前标签为0，则将标签更改为0
    """

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


    print("加载预处理后的数据")
    all_files = glob.glob(f"./data/*encoded.csv")
    
    # 读取并合并所有文件
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)

       
    encoded_df = pd.concat(df_list, ignore_index=True)


 

    

    
   
    
       # 创建要处理的文件名列表
    file_patterns = [
        # "from_Africa_Africa_encoded.csv",
        # "from_Central America_Central America_encoded.csv",
        # "from_Central America_North America_encoded.csv",
        # "from_East Asia_East Asia_encoded.csv",
        "from_Europe_Europe_encoded.csv",
        # "from_Europe_West Asia_encoded.csv",
        # "from_North America_Central America_encoded.csv",
        # "from_North America_North America_encoded.csv",
        # "from_Oceania_Oceania_encoded.csv",
        # "from_South America_South America_encoded.csv",
        # "from_Southeast Asia_Southeast Asia_encoded.csv",
        # "from_West Asia_Europe_encoded.csv",
        # "from_West Asia_West Asia_encoded.csv",
        "others_encoded.csv"
    ]
    
    # 创建一个字典来存储特定文件的采样率（如果有不同的采样率）,对美洲区域少采样一点嘛
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
        
        sample_frac = sample_rates.get(file_pattern, 1.0)
        
        df = prepare_data(
            data_dir="./data/", 
            fintune_pattern=file_pattern, 
            encoded_df=encoded_df,
            all_files=all_files,
            sample_frac=sample_frac, 
            random_state=42,
            fintune=fintune
        )
        if df is None:
            print(f"警告：prepare_data 返回 None，跳过文件 {file_pattern}")
            continue
       
        dfs.append(df)
        
        
        print(f"处理完成: {file_pattern}, 采样率: {sample_frac}, 行数: {len(df)}")
    
    
    encoded_df = pd.concat(dfs, ignore_index=True)
    
    encoded_df = encoded_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"合并后的数据集大小: {len(encoded_df)}条记录")

    
    

    inputs = get_input(encoded_df)  # 获取输入特征


    

    label1 = encoded_df['label'].values # CTR标签 (二分类),航司预测
    label2 = encoded_df['aircraft_model'].values # 机型标签 (多分类)


     # 合并标签用于划分数据集
    combined_labels = {'ctr_output': label1, 'aircraft_output': label2}

    # 划分训练集、验证集和测试集
    temp_indices, test_indices = train_test_split(
        range(len(label1)), test_size=0.2, random_state=42, stratify=label1
    )

    train_indices, val_indices = train_test_split(
        temp_indices, test_size=0.25, random_state=42, 
        stratify=label1[temp_indices]
    )

    # 准备训练集、验证集和测试集的输入和标签
    train_inputs = {k: v[train_indices] for k, v in inputs.items()}
    val_inputs = {k: v[val_indices] for k, v in inputs.items()}
    test_inputs = {k: v[test_indices] for k, v in inputs.items()}

    train_labels = {
        'ctr_output': label1[train_indices].astype(np.float32),
        'aircraft_output': label2[train_indices].astype(np.int32)
    }
    val_labels = {
        'ctr_output': label1[val_indices].astype(np.float32),
        'aircraft_output': label2[val_indices].astype(np.int32)
    }
    test_labels = {
        'ctr_output': label1[test_indices].astype(np.float32),
        'aircraft_output': label2[test_indices].astype(np.int32)
    }

    print(f"训练集大小: {len(train_indices)}")
    print(f"验证集大小: {len(val_indices)}")
    print(f"测试集大小: {len(test_indices)}")
    print(f"CTR正样本比例: {np.mean(train_labels['ctr_output']):.4f}")

 


    # 用dataset管理数据集，避免显存溢出
    def create_dataset(inputs, labels, batch_size, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, len(list(labels.values())[0])))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_dataset(train_inputs, train_labels, batch_size, shuffle=False)
    val_dataset = create_dataset(val_inputs, val_labels, batch_size, shuffle=False)
    test_dataset = create_dataset(test_inputs, test_labels, batch_size, shuffle=False)

    del encoded_df  # 删除df节省内存
    





   
    class ReinforcementRuleCallback(tf.keras.callbacks.Callback):
        def __init__(self, train_dataset, train_inputs, train_labels):
            super(ReinforcementRuleCallback, self).__init__()
            self.train_dataset = train_dataset
            self.train_inputs = train_inputs
            self.train_labels = train_labels
            self.modified_indices = []
        
        def on_epoch_end(self, epoch, logs=None):
        
            print("\n应用强化规则...")
            
            # 获取训练数据的预测结果
            predictions = self.model.predict(self.train_inputs)
            ctr_predictions = predictions['ctr_output']  # 获取CTR预测结果
            
            # 追踪修改的样本数量
            modified_count = 0
            
            # 创建原始数据的副本进行修改
            modified_train_ctr = np.copy(self.train_labels['ctr_output'])
            
            # 收集特殊样本索引
            special_indices = []
            
            for i in range(len(self.train_labels['ctr_output'])):
                # 检查预测为正样本,即预测为1，实际应该为0的样本
                if ctr_predictions[i] > 0.5:
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
                    if not (is_dep_in_main_areas or is_arr_in_main_areas) and modified_train_ctr[i] == 0:
                        # 将标签更改为0（实际上不需要改变，因为已经是0）
                        modified_train_ctr[i] = 0
                        modified_count += 1
                        self.modified_indices.append(i)
                        special_indices.append(i)
            
            if modified_count > 0:
                print(f"根据强化规则发现了{modified_count}个样本")
                
                # 收集标签为1但预测概率小于0.5的样本（模型误判的正样本）
                misclassified_positives = [i for i in range(len(self.train_labels['ctr_output'])) 
                                        if self.train_labels['ctr_output'][i] == 1 and ctr_predictions[i] < 0.5]
                
                print(f"发现{len(misclassified_positives)}个被误判的正样本")
                
                # 查找所有真实标签为1的样本
                positive_indices = [i for i in range(len(self.train_labels['ctr_output'])) 
                                  if self.train_labels['ctr_output'][i] == 1]
                
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
                balanced_labels = {
                    'ctr_output': self.train_labels['ctr_output'][balanced_indices],
                    'aircraft_output': self.train_labels['aircraft_output'][balanced_indices]
                }
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
                self.train_labels['ctr_output'] = modified_train_ctr
                
                # 使用更新后的标签重新创建主训练数据集
                self.train_dataset = create_dataset(
                    self.train_inputs, 
                    self.train_labels, 
                    batch_size, 
                    shuffle=True
                )
                
                print(f"强化规则应用完成。到目前为止共发现了{len(set(self.modified_indices))}个样本")
            else:
                print("未发现符合条件的样本，跳过强化训练")
                
    reinforcement_callback = ReinforcementRuleCallback(
        train_dataset=train_dataset,
        train_inputs=train_inputs,
        train_labels=train_labels
    )

   
    

 

    callbacks = [
        TrainingMonitor(),
        reinforcement_callback  
    ]

    
  
    batch_size=batch_size
    epochs=epochs
    
    
  

    learning_rate=0.01
    # 使用带调度的优化器
    optimizer = Adam(learning_rate=learning_rate)
    
    # 编译模型 - 多任务学习配置
    model.compile(
        optimizer=optimizer,
        loss={
            'ctr_output': 'binary_crossentropy',  # CTR任务用二元交叉熵
            'aircraft_output': 'sparse_categorical_crossentropy'  # 机型预测用稀疏分类交叉熵
        },
        loss_weights={
            'ctr_output': 1.0,  # CTR任务权重
            'aircraft_output': 1.0  # 机型预测任务权重
        },
        metrics={
            'ctr_output': ['accuracy', 'precision', 'recall', 'auc'],
            'aircraft_output': ['accuracy', 'sparse_top_k_categorical_accuracy']
        }
    )


 
    # 8. 训练模型
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        verbose=1,
        callbacks=callbacks
    )

    print("保存模型...")
    model.save(model_path)
    print(f"模型已保存为 {model_path}")

    






if __name__ == "__main__":
    model_path="./models/mmoe_final_model.keras"
    
    ##拿到各个特征的大小,准备模型构造
    emmbedding_size_df = pd.read_csv(r"./data/embedding_size.csv")
    print("特征嵌入大小：", emmbedding_size_df)
    icao_size = emmbedding_size_df['icao_size'].values[0]
    city_size = emmbedding_size_df['city_size'].values[0]
    country_size = emmbedding_size_df['country_size'].values[0]
    area_size = emmbedding_size_df['area_size'].values[0]
    supplier_size = emmbedding_size_df['supplier_size'].values[0]
    aircraft_model_size = emmbedding_size_df['aircraft_model_size'].values[0]
    aircraft_scale_size = emmbedding_size_df['aircraft_scale_size'].values[0]
    flight_distance_size = emmbedding_size_df['flight_distance_size'].values[0]
    print(f"icao_size: {icao_size}, city_size: {city_size}, country_size: {country_size}, area_size: {area_size}, supplier_size: {supplier_size}, aircraft_model_size: {aircraft_model_size}, aircraft_scale_size: {aircraft_scale_size}, flight_distance_size: {flight_distance_size}")

   

    #创建模型，
    embedding_dim_input = 8  # 嵌入维度
    num_heads_input = 2  # 注意力头数
    gate_units_input = (8, 4)  # 门控网络的隐藏层单元数
    expert_units_input = (12, 8, 4)  # 专家网络
    num_expert_input = 3  # 专家数量
    num_task_input = 2  # 任务数量
    task_tower_units_input = (6, 2)  # 任务塔的隐藏层单元数
    depth_attention_input = 2  # 注意力层的深度
    model = create_MMOE(
        icao_size=icao_size,
        city_size=city_size,
        country_size=country_size,
        area_size=area_size,
        supplier_size=supplier_size,
        aircraft_model_size=aircraft_model_size,
        aircraft_scale_size=aircraft_scale_size,
        flight_distance_size=flight_distance_size,
        embedding_dim=embedding_dim_input,
        num_heads=num_heads_input,
        gate_units=gate_units_input,
        expert_units=expert_units_input,
        num_expert=num_expert_input,
        num_task=num_task_input,
        task_tower_units=task_tower_units_input,
        depth_attention=depth_attention_input
    )


    model.summary()
    
    MMOE_train_real(batch_size=12,epochs=2,model_path=model_path,model=model,fintune_pattern="*Asia*encoded.csv",fintune=True)
        

  


    


    








