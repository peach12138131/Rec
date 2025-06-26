"""一个测试模型"""
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
import pickle
import pandas as pd
import numpy as np
from blocks import ResidualBlock, FeatureAttention, CrossLayer,DINLayer
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, roc_auc_score
import time
try:
    from tensorflow.keras.saving import register_keras_serializable
except:
    # 对于较旧版本的TensorFlow
    from tensorflow.keras.utils import register_keras_serializable




# 添加这个装饰器来注册模型
@register_keras_serializable()
class MMOE(Model):
    def __init__(self, icao_size, city_size, country_size, area_size, supplier_size,aircraft_model_size,aircraft_scale_size ,flight_distance_size,aircraft_model_size_predict,
                 embedding_dim=8, num_heads=2, gate_units=(80,40), expert_units=(512,256,128),num_expert=3,num_task=2,task_tower_units=(256,128),depth_attention=2,**kwargs):
        super(MMOE, self).__init__(**kwargs)
        
        self.icao_size = icao_size 
        self.city_size = city_size 
        self.country_size = country_size 
        self.area_size = area_size
        self.supplier_size = supplier_size
        self.aircraft_model_size = aircraft_model_size
        self.aircraft_scale_size =aircraft_scale_size 
        self.flight_distance_size = flight_distance_size
        self.aircraft_model_size_predict=aircraft_model_size_predict  # 预测的机型数量



        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.depth_attention = depth_attention
    
        self.expert_units = expert_units
        self.gate_units = gate_units
        self.num_expert = num_expert
        self.num_task = num_task
        
        


         # Embedding层
        self.supplier_embedding = layers.Embedding(
            self.supplier_size, 
            embedding_dim,
            name='supplier_embedding'
        )
        self.area_embedding = layers.Embedding(
            self.area_size,
            embedding_dim,
            name='supplier_area_embedding'
        )
        self.aircraft_model_embedding = layers.Embedding(
            self.aircraft_model_size,
            embedding_dim,
            name='supplier_aircraft_model_embedding'
        )
        self.country_embedding = layers.Embedding(
            self.country_size,
            embedding_dim,
            name='supplier_country_embedding'
        )
        self.icao_embedding = layers.Embedding(
            self.icao_size,  
            embedding_dim,
            name='supplier_icao_embedding'
        )
        self.city_embedding = layers.Embedding(
            self.city_size,  
            embedding_dim,
            name='supplier_city_embedding'
        )
        self.aircraft_scale_embedding = layers.Embedding(
            self.aircraft_scale_size,  
            embedding_dim,
            name='aircraft_scale_embedding'
        )
        self.flight_distance_embedding = layers.Embedding(
            self.flight_distance_size,  
            embedding_dim,
            name='flight_distance_embedding'
        )


        #transformer层
        self.attention_blocks = [FeatureAttention(embedding_dim=embedding_dim, num_heads=num_heads) for _ in range(self.depth_attention)]

        self.din_layer1= DINLayer()
        self.din_layer2= DINLayer()
        self.din_layer3= DINLayer()
        self.din_layer4= DINLayer()

        #flatten层和dense层
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(
                            512,
                            activation='relu',
                            name='dense1',
                        )
        self.dropout1 = layers.Dropout(0.2)

        # MMOE 专家网络
        self.experts = []
        for i in range(self.num_expert):
            expert_layers = []
            for j, units in enumerate(self.expert_units):
                expert_layers.append(
                    layers.Dense(
                        units, 
                        activation='relu', 
                        name=f'expert_{i}_layer_{j}'
                    )
                )
            self.experts.append(expert_layers)


        self.gates = []  #多少任务多少门
        for i in range(self.num_task):
            gate_layers = []
            for j, units in enumerate(self.gate_units):
                gate_layers.append(
                    layers.Dense(
                        units,
                        activation='relu',
                        name=f'gate_{i}_layer_{j}'
                    )
                )
            # 门控网络的输出层，输出专家权重
            gate_layers.append(
                layers.Dense(
                    self.num_expert,  # 输出维度等于专家数量
                    activation='softmax',  # softmax确保权重和为1
                    name=f'gate_{i}_output'
                )
            )
            self.gates.append(gate_layers)



        #输出任务塔定义
        self.task_tower=[]
        for i in range(self.num_task):
            task_layers = []
            for j, units in enumerate(task_tower_units):
                task_layers.append(
                    layers.Dense(
                        units,
                        activation='relu',
                        name=f'task_{i}_layer_{j}'
                    )
                )
            self.task_tower.append(task_layers)

        self.task_tower[0].append(
            layers.Dense(
                1,
                activation='sigmoid',  # sigmoid用于二分类
                name='tower_ctr_output'
            )
        ) 

        self.task_tower[1].append(
            layers.Dense(
                self.aircraft_model_size_predict,  # 输出维度等于机型数量
                activation='softmax',      # softmax用于多分类
                name='tower_aircraft_output'
            )
        )
        

        
       

    def build(self, input_shape):
        # 创建一个输入来构建子层
        batch_size = 1
        dummy_inputs = {
            'dep_icao': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_icao': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_city_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_city_name': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_area_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_area_name': tf.zeros((batch_size,), dtype=tf.int32),
            'supplier_id': tf.zeros((batch_size,), dtype=tf.int32),
            'supplier_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'supplier_aircraft_scale_type': tf.zeros((batch_size,), dtype=tf.int32),
            'flight_distance_bin': tf.zeros((batch_size,), dtype=tf.int32),
        }

        # 添加主要区域特征
        for i in range(1, 4):
            dummy_inputs[f'main_area_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)
    
        # 添加机型特征 (修正为30个)
        for i in range(1, 31):
            dummy_inputs[f'aircraft_model_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)

        # 添加基地特征
        for i in range(1, 7):
            dummy_inputs[f'base_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)

        # 添加基地城市特征
        for i in range(1, 7):
            dummy_inputs[f'base_city_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)
        
        super().build(input_shape)
        print("\033[93m input_shape:", input_shape, "\033[0m")
      
        

    def compute_output_shape(self, input_shapes):
        batch_size = list(input_shapes.values())[0][0]
        return [(batch_size,)]
    


    def process_input_and_embedding(self, inputs):
        supplier_emb = self.supplier_embedding(inputs['supplier_id'])
        supplier_country_emb = self.country_embedding(inputs['supplier_country_name'])
        aircraft_scale_emb = self.aircraft_scale_embedding(inputs['supplier_aircraft_scale_type'])  

        # 循环共享area,embedding
        area_embs = [self.area_embedding(inputs[f'main_area_{i}']) for i in range(1, 4)]  ##3个地区

      
        aircraft_model_embs = [self.aircraft_model_embedding(inputs[f'aircraft_model_{i}']) for i in range(1, 31)]  ##30个,返回python列表

        base_embs = [self.icao_embedding(inputs[f'base_{i}']) for i in range(1, 7)]  ##6个base,和icao共享嵌入

        city_embs = [self.city_embedding(inputs[f'base_city_{i}']) for i in range(1, 7)]  ##6个base_city,

        dep_icao_emb = self.icao_embedding(inputs['dep_icao'])
        arr_icao_emb = self.icao_embedding(inputs['arr_icao'])
        dep_city_emb = self.city_embedding(inputs['dep_city_name'])
        arr_city_emb = self.city_embedding(inputs['arr_city_name'])
     
        dep_country_emb = self.country_embedding(inputs['dep_country_name'])
        arr_country_emb = self.country_embedding(inputs['arr_country_name'])
        dep_area_emb = self.area_embedding(inputs['dep_area_name']) 
        arr_area_emb = self.area_embedding(inputs['arr_area_name'])
        flight_distance_emb = self.flight_distance_embedding(inputs['flight_distance_bin'])

        return supplier_emb, supplier_country_emb, aircraft_scale_emb, area_embs, aircraft_model_embs, base_embs, city_embs, dep_icao_emb, arr_icao_emb, dep_city_emb, arr_city_emb, dep_country_emb, arr_country_emb, dep_area_emb, arr_area_emb, flight_distance_emb
    
    def call(self, inputs, training=False):
        supplier_emb, supplier_country_emb, aircraft_scale_emb, area_embs, aircraft_model_embs, base_embs, city_embs, dep_icao_emb, arr_icao_emb, dep_city_emb, arr_city_emb, dep_country_emb, arr_country_emb, dep_area_emb, arr_area_emb, flight_distance_emb = self.process_input_and_embedding(inputs)

        airport_feature = dep_icao_emb + arr_icao_emb + dep_city_emb + arr_city_emb + dep_country_emb + arr_country_emb + dep_area_emb + arr_area_emb + flight_distance_emb

        # DIN层调用：将embedding列表堆叠成序列张量
        area_keys = tf.stack(area_embs, axis=1)  # [batch_size, 3, embedding_dim]
        aircraft_model_keys = tf.stack(aircraft_model_embs, axis=1)  # [batch_size, 30, embedding_dim]  
        base_keys = tf.stack(base_embs, axis=1)  # [batch_size, 6, embedding_dim]
        city_keys = tf.stack(city_embs, axis=1)  # [batch_size, 6, embedding_dim]

        # DIN层期望的格式：[query, keys]
        supplier_area_pool = self.din_layer1([airport_feature, area_keys], training=training)
        aircraft_model_pool = self.din_layer2([airport_feature, aircraft_model_keys], training=training)
        base_pool = self.din_layer3([airport_feature, base_keys], training=training)
        city_pool = self.din_layer4([airport_feature, city_keys], training=training)

        x = tf.stack([
            dep_icao_emb,
            arr_icao_emb,
            flight_distance_emb,
            airport_feature,
            supplier_emb,
            supplier_country_emb,
            aircraft_scale_emb,
            supplier_area_pool,
            aircraft_model_pool,
            base_pool,
            city_pool    
        ], axis=1)

        # transformer 卷一下
        for attention_block in self.attention_blocks:
            x, _ = attention_block(x, training=training)

        
        x = self.flatten(x)  # 展平进入后续
        x=self.dense1(x)  # 经过一个全连接层,映射到512维度
        x = self.dropout1(x, training=training)  # 应用dropout
        
        # MMOE核心部分：专家网络处理
        expert_outputs = []
        for i, expert_layers in enumerate(self.experts):
            expert_output = x
            for layer in expert_layers:
                expert_output = layer(expert_output, training=training)
            expert_outputs.append(expert_output)
        
        # 将专家输出堆叠: [batch_size, num_expert, expert_output_dim]
        expert_outputs = tf.stack(expert_outputs, axis=1)
        
        # 门控网络为每个任务生成专家权重
        task_inputs = []
        for i, gate_layers in enumerate(self.gates):
            gate_output = x
            for layer in gate_layers:
                gate_output = layer(gate_output, training=training)
            
            # gate_output shape: [batch_size, num_expert]
            # expert_outputs shape: [batch_size, num_expert, expert_output_dim]
            # 使用权重对专家输出进行加权平均
            gate_output = tf.expand_dims(gate_output, axis=2)  # [batch_size, num_expert, 1]
            weighted_expert_output = tf.reduce_sum(
                expert_outputs * gate_output, axis=1
            )  # [batch_size, expert_output_dim]
            
            task_inputs.append(weighted_expert_output)
        
        # 任务塔处理
        task_outputs = []
        for i, (task_layers, task_input) in enumerate(zip(self.task_tower, task_inputs)):
            task_output = task_input
            for layer in task_layers:
                task_output = layer(task_output, training=training)
            task_outputs.append(task_output)
        
        # 返回两个任务的输出
        ctr_output = task_outputs[0]  # CTR预测输出 [batch_size, 1]
        aircraft_output = task_outputs[1]  # 机型预测输出 [batch_size, aircraft_model_size]
        
        return {
            'ctr_output': ctr_output,
            'aircraft_output': aircraft_output
        }
    
    def get_config(self):
        config = super(MMOE, self).get_config()
        config.update({
            'icao_size': self.icao_size ,
            'city_size': self.city_size ,
            'country_size': self.country_size ,
            'area_size': self.area_size ,
            'supplier_size': self.supplier_size ,
            'aircraft_model_size': self.aircraft_model_size ,
            'aircraft_scale_size': self.aircraft_scale_size ,
            'flight_distance_size': self.flight_distance_size ,
            'aircraft_model_size_predict': self.aircraft_model_size_predict,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'gate_units': self.gate_units,
            'expert_units': self.expert_units,
            'num_expert': self.num_expert,
            'num_task': self.num_task,
            'depth_attention': self.depth_attention,
        })
        return config



def create_MMOE(icao_size, city_size, country_size, area_size, supplier_size,aircraft_model_size,aircraft_scale_size ,flight_distance_size,
                 embedding_dim=8, num_heads=2, gate_units=(80,40), expert_units=(512,256,128),num_expert=3,num_task=2,task_tower_units=(256,128),depth_attention=2,):
    """
    创建模型
    
    参数:
    city_vocab_size: 城市词汇表大小 = len(city_encoder.classes_)
    region_vocab_size: 地区词汇表大小 = len(region_encoder.classes_)
    supplier_vocab_size: 航司词汇表大小 = len(supplier_encoder.classes_)
    embedding_dim: embedding维度，默认8
    num_heads: 注意力头数，默认2
    """
    model = MMOE(
        icao_size=icao_size+2,
        city_size=city_size+2,
        country_size=country_size+2,
        area_size=area_size+2,
        supplier_size=supplier_size+2,
        aircraft_model_size=aircraft_model_size+2,
        aircraft_scale_size=aircraft_scale_size+2,
        flight_distance_size=flight_distance_size+2,
        aircraft_model_size_predict = aircraft_model_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        gate_units=gate_units,
        expert_units=expert_units,
        num_expert=num_expert,
        num_task=num_task,
        task_tower_units=task_tower_units,
        depth_attention=depth_attention
    )
    
     # 创建一个批次的示例输入来构建模型，结构相同，用于第一次初始化
    batch_size = 1
    dummy_inputs = {
            'dep_icao': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_icao': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_city_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_city_name': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_area_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_area_name': tf.zeros((batch_size,), dtype=tf.int32),
            'supplier_id': tf.zeros((batch_size,), dtype=tf.int32),
            'supplier_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'supplier_aircraft_scale_type': tf.zeros((batch_size,), dtype=tf.int32),
            'flight_distance_bin': tf.zeros((batch_size,), dtype=tf.int32),
        }

    # 添加主要区域特征
    for i in range(1, 4):
        dummy_inputs[f'main_area_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)

    # 添加机型特征 (修正为30个)
    for i in range(1, 31):
        dummy_inputs[f'aircraft_model_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)

    # 添加基地特征
    for i in range(1, 7):
        dummy_inputs[f'base_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)

    # 添加基地城市特征
    for i in range(1, 7):
        dummy_inputs[f'base_city_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)
    
    # 进行一次前向传播来构建模型
    _ = model(dummy_inputs)
    
    print("\033[33m" + f"MMOE model outputs:" + "\033[0m")
    for key, value in _.items():
        print(f"  {key}: {value.shape}")
    return model



def get_input(encoded_df):
    #准备输入数据
    inputs={
        
        'dep_icao': encoded_df['dep_icao'].values,
        'arr_icao': encoded_df['arr_icao'].values,
        'dep_city_name': encoded_df['dep_city_name'].values,
        'arr_city_name': encoded_df['arr_city_name'].values,
        'dep_country_name': encoded_df['dep_country_name'].values,
        'arr_country_name': encoded_df['arr_country_name'].values,
        'dep_area_name': encoded_df['dep_area_name'].values,
        'arr_area_name': encoded_df['arr_area_name'].values,
        'supplier_id': encoded_df['supplier_id'].values,
        'supplier_country_name': encoded_df['supplier_country_name'].values,
        'supplier_aircraft_scale_type':encoded_df['supplier_aircraft_scale_type'].values,
        'flight_distance_bin': encoded_df['flight_distance_bin'].values,
    }


    for i in range(1, 4):
        inputs[f'main_area_{i}'] = encoded_df[f'main_area_{i}'].values

    # 添加机型特征 (30个机型)
    for i in range(1, 31):
        inputs[f'aircraft_model_{i}'] = encoded_df[f'aircraft_model_{i}'].values

    for i in range(1, 7):
        inputs[f'base_{i}'] = encoded_df[f'base_{i}'].values

    for i in range(1, 7):
        inputs[f'base_city_{i}'] = encoded_df[f'base_city_{i}'].values


    return inputs





def MMOE_train(batch_size,epochs):
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

      

     # 加载预处理后的数据,再随机打乱
    df1=pd.read_csv(r"./data/others_encoded.csv")
   
    encoded_df = df1
    encoded_df = encoded_df.sample(frac=1).reset_index(drop=True)

    inputs = get_input(encoded_df)  # 准备输入数据

   


    

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
    def create_dataset(inputs, labels, batch_size, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, len(list(labels.values())[0])))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_dataset(train_inputs, train_labels, batch_size, shuffle=False)
    val_dataset = create_dataset(val_inputs, val_labels, batch_size, shuffle=False)
    test_dataset = create_dataset(test_inputs, test_labels, batch_size, shuffle=False)

    del encoded_df, df1  # 删除df节省内存
    







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

   

    #创建模型，测试而已，不用创建很大，小一些
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
    learning_rate=0.01
    # 编译模型 - 多任务学习配置
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'ctr_output': 'binary_crossentropy',  # CTR任务用二元交叉熵
            'aircraft_output': 'sparse_categorical_crossentropy'  # 机型预测用稀疏分类交叉熵
        },
        loss_weights={
            'ctr_output': 1.0,  # CTR任务权重
            'aircraft_output': 1  # 机型预测任务权重
        },
        metrics={
            'ctr_output': ['accuracy', 'precision', 'recall'],
            'aircraft_output': ['accuracy', 'sparse_top_k_categorical_accuracy']
        }
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
    ]

    # 训练模型
    print("开始训练...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
     
    print("保存模型...")
    model.save('./models/mmoe_final_model.keras')
    print("模型已保存至: ./models/mmoe_final_model.keras")

    # 评估模型
    print("\n开始评估...")
    
    # 在测试集上进行预测
    test_predictions = model.predict(test_dataset, verbose=1)
    ctr_pred = test_predictions['ctr_output'].flatten()
    aircraft_pred = test_predictions['aircraft_output']
    
    # CTR任务评估
    ctr_pred_binary = (ctr_pred > 0.5).astype(int)
    ctr_true = test_labels['ctr_output']
    
    ctr_accuracy = accuracy_score(ctr_true, ctr_pred_binary)
    ctr_precision = precision_score(ctr_true, ctr_pred_binary, average='binary')
    ctr_recall = recall_score(ctr_true, ctr_pred_binary, average='binary')
    ctr_f1 = f1_score(ctr_true, ctr_pred_binary, average='binary')
    
    try:
        ctr_auc = roc_auc_score(ctr_true, ctr_pred)
    except:
        ctr_auc = 0.0
    
    print(f"\nCTR任务评估结果:")
    print(f"Accuracy: {ctr_accuracy:.4f}")
    print(f"Precision: {ctr_precision:.4f}")
    print(f"Recall: {ctr_recall:.4f}")
    print(f"F1-Score: {ctr_f1:.4f}")
    print(f"AUC: {ctr_auc:.4f}")
    
    # 机型预测任务评估
    aircraft_pred_class = np.argmax(aircraft_pred, axis=1)
    aircraft_true = test_labels['aircraft_output']
    
    aircraft_accuracy = accuracy_score(aircraft_true, aircraft_pred_class)
    aircraft_precision = precision_score(aircraft_true, aircraft_pred_class, average='weighted', zero_division=0)
    aircraft_recall = recall_score(aircraft_true, aircraft_pred_class, average='weighted', zero_division=0)
    aircraft_f1 = f1_score(aircraft_true, aircraft_pred_class, average='weighted', zero_division=0)
    
    # Top-5准确率
    top5_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(
        aircraft_true, aircraft_pred, k=5
    ).numpy().mean()
    
    print(f"\n机型预测任务评估结果:")
    print(f"Accuracy: {aircraft_accuracy:.4f}")
    print(f"Precision: {aircraft_precision:.4f}")
    print(f"Recall: {aircraft_recall:.4f}")
    print(f"F1-Score: {aircraft_f1:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    
    # 保存训练历史
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # CTR损失
    plt.subplot(2, 3, 2)
    plt.plot(history.history['ctr_output_loss'], label='Train CTR Loss')
    plt.plot(history.history['val_ctr_output_loss'], label='Val CTR Loss')
    plt.title('CTR Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 机型预测损失
    plt.subplot(2, 3, 3)
    plt.plot(history.history['aircraft_output_loss'], label='Train Aircraft Loss')
    plt.plot(history.history['val_aircraft_output_loss'], label='Val Aircraft Loss')
    plt.title('Aircraft Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # CTR准确率
    plt.subplot(2, 3, 4)
    plt.plot(history.history['ctr_output_accuracy'], label='Train CTR Acc')
    plt.plot(history.history['val_ctr_output_accuracy'], label='Val CTR Acc')
    plt.title('CTR Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 机型预测准确率
    plt.subplot(2, 3, 5)
    plt.plot(history.history['aircraft_output_accuracy'], label='Train Aircraft Acc')
    plt.plot(history.history['val_aircraft_output_accuracy'], label='Val Aircraft Acc')
    plt.title('Aircraft Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 学习率
    plt.subplot(2, 3, 6)
    if 'lr' in history.history:
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('./results/mmoe_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 返回评估结果
    results = {
        'ctr_metrics': {
            'accuracy': ctr_accuracy,
            'precision': ctr_precision,
            'recall': ctr_recall,
            'f1': ctr_f1,
            'auc': ctr_auc
        },
        'aircraft_metrics': {
            'accuracy': aircraft_accuracy,
            'precision': aircraft_precision,
            'recall': aircraft_recall,
            'f1': aircraft_f1,
            'top5_accuracy': top5_accuracy
        },
        'model': model,
        'history': history
    }
    
    return results


   



if __name__ == "__main__":
    
    import os
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # 开始训练
    results = MMOE_train(batch_size=12, epochs=2)
    
    print("\n训练完成！")
    print("最佳模型已保存至: ./models/mmoe_best_model.h5")
    print("训练历史图表已保存至: ./results/mmoe_training_history.png")