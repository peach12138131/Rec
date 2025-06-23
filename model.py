"""一个测试模型，维度为8,多头为2"""
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
import pickle
import pandas as pd
import numpy as np
from blocks import ResidualBlock, FeatureAttention, CrossLayer
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
class SupplierTower(layers.Layer):
    def __init__(self, icao_size, city_size,supplier_size,country_size,area_size,aircraft_model_size, embedding_dim=8, num_heads=2, output_dim=8, units=256, depth_mlp=2,depth_attention=2, **kwargs):
        """航司塔"""
        super(SupplierTower, self).__init__(**kwargs)
        
        # 保存参数以便在 get_config 中使用
        self.supplier_size = supplier_size
        self.country_size = country_size
        self.area_size = area_size
        self.aircraft_model_size = aircraft_model_size
        self.icao_size = icao_size
        self.city_size = city_size

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.units = units
        self.depth_mlp = depth_mlp
        self.depth_attention = depth_attention

        # Embedding层
        self.supplier_embedding = layers.Embedding(
            supplier_size, 
            embedding_dim,
            name='supplier_embedding'
        )
        self.area_embedding = layers.Embedding(
            area_size,
            embedding_dim,
            name='supplier_area_embedding'
        )
        self.aircraft_model_embedding = layers.Embedding(
            aircraft_model_size,
            embedding_dim,
            name='supplier_aircraft_model_embedding'
        )
        self.country_embedding = layers.Embedding(
            country_size,
            embedding_dim,
            name='supplier_country_embedding'
        )
        self.icao_embedding = layers.Embedding(
            icao_size,
            embedding_dim,
            name='supplier_icao_embedding'
        )
        self.city_embedding = layers.Embedding(
            city_size,
            embedding_dim,
            name='supplier_city_embedding'
        )

        self.aircraft_scale_embedding = layers.Dense(embedding_dim,  name='supplier_aircraft_scale_embedding')  ##虽然名字叫，但是浮点数值用线性层代替embeding




        
        # 注意力层
        self.attention_blocks = [FeatureAttention(embedding_dim=embedding_dim, num_heads=num_heads) for _ in range(self.depth_attention)]

        # 展平层
        self.flatten = layers.Flatten()
        
         # 修改为动态创建残差网络层
        self.residual_blocks = [ResidualBlock(units=self.units) for _ in range(self.depth_mlp)]
        self.dcn_blocks = [CrossLayer() for _ in range(self.depth_mlp)]
   
        
        # 网络融合层
        self.fusion = layers.Dense(embedding_dim)
        self.shourcut_fusion = layers.Dense(embedding_dim)
        
        # 输出层，避免在call中创建
        self.output_dense = layers.Dense(self.output_dim)

    def build(self, input_shape):
        super().build(input_shape)
        print("\033[93msupplierTower input_shape:", input_shape, "\033[0m")

    def compute_output_shape(self, input_shape):
        return (input_shape['supplier_id'][0], self.output_dim)
    

    def process_input_and_embedding(self,inputs):
        supplier_emb = self.supplier_embedding(inputs['supplier_id'])
        supplier_country_emb = self.country_embedding(inputs['supplier_country_name'])
        aircraft_scale = tf.reshape(inputs['supplier_aircraft_scale_type'], [-1, 1])
        aircraft_scale_emb = self.aircraft_scale_embedding(aircraft_scale)

        #循环共享area,embedding
        area_embs = [self.area_embedding(inputs[f'main_area_{i}']) for i in range(1, 4)]  ##3个地区

        aircraft_model_emb = [self.aircraft_model_embedding(inputs[f'aircraft_model_{i}']) for i in range(1, 16)]  ##15个机型

        base_embs = [self.icao_embedding(inputs[f'base_{i}']) for i in range(1, 7)]  ##6个base,和icao共享嵌入

        city_embs = [self.city_embedding(inputs[f'base_city_{i}']) for i in range(1, 7)]  ##6个base_city,

        x = tf.stack([
            supplier_emb, 
            supplier_country_emb, 
            aircraft_scale_emb,
            *area_embs,
            *aircraft_model_emb,
            # *base_embs,
            # *city_embs
        ], axis=1)

        return x
    
    
   
    # 定义模型前向传播过程,拼接各个模块
    def call(self, inputs, training=False):
       

        x=self.process_input_and_embedding(inputs)


        # 注意力层
        for attention_block in self.attention_blocks:
            x, _ = attention_block(x, training=training)
 

        # 直接展平
        x = self.flatten(x)  # (batch_size, num_features * embedding_dim) 
        
        x0=x
         # 使用循环处理残差网络
        res_out = x
        for residual_block in self.residual_blocks:
            res_out = residual_block(res_out, training=training)
            
        dcn_out = x
        for dcn_block in self.dcn_blocks:
            dcn_out = dcn_block(dcn_out, training=training)
        
        # 融合网络输出
        fusion = tf.concat([res_out, dcn_out], axis=-1)
        out = self.fusion(fusion)
        x0 = self.shourcut_fusion(x0)  ##x0残差链接
        out=out+x0
        
        # 调整输出维度为 output_dim
        out = self.output_dense(out)
        
       
        
     
        
        return out
    
    def get_config(self):
        config = super(SupplierTower, self).get_config()
        config.update({
            'suppl  ier_size': self.supplier_size,
            'icao_size': self.icao_size,
            'city_size': self.city_size,
            'country_size': self.country_size,
            'area_size': self.area_size,
            'aircraft_model_size': self.aircraft_model_size,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'output_dim': self.output_dim,
            'units': self.units,
            'depth_mlp': self.depth_mlp,
            'depth_attention': self.depth_attention
        })
        return config


# 添加这个装饰器来注册模型
@register_keras_serializable()
class CityTower(layers.Layer):
    def __init__(self, icao_size,city_size, country_size,area_size ,embedding_dim=8, num_heads=2, output_dim=8, units=256, depth_mlp=2,depth_attention=2, **kwargs):
        """城市塔"""
        super(CityTower, self).__init__(**kwargs)
        
        # 保存参数以便在 get_config 中使用，vocab_size是类别数目,来自于自然语言处理，这里就是机场的类别数目
        self.icao_size = icao_size
        self.city_size = city_size
        self.country_size = country_size
        self.area_size = area_size


        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.units = units
        self.depth_mlp = depth_mlp
        self.depth_attention = depth_attention

        # Embedding层
        self.icao_embedding = layers.Embedding(
            icao_size, 
            embedding_dim,
            name='icao_embedding'
        )
        self.city_embedding = layers.Embedding(
            city_size,
            embedding_dim,
            name='city_embedding'
        )
        self.country_embeding=layers.Embedding(
            country_size,
            embedding_dim,
            name='country_embedding'
        )
        self.area_embeding=layers.Embedding(
            area_size,
            embedding_dim,
            name='area_embedding'
        )
        




        # 中间塔
        self.attention_blocks = [FeatureAttention(embedding_dim=embedding_dim, num_heads=num_heads) for _ in range(self.depth_attention)]
        # 展平层
        self.flatten = layers.Flatten()
        self.residual_blocks = [ResidualBlock(units=self.units) for _ in range(self.depth_mlp)]
        self.dcn_blocks = [CrossLayer() for _ in range(self.depth_mlp)]
        self.fusion = layers.Dense(embedding_dim)# 展平层
        self.flatten = layers.Flatten()
        self.shourcut_fusion = layers.Dense(embedding_dim)
        
        
        
        # 最终融合
        self.final_fusion = layers.Dense(embedding_dim)
        
        # 输出层，避免在call中创建
        self.output_dense = layers.Dense(self.output_dim)

        

    def build(self, input_shape):
        super().build(input_shape)
        print("\033[93mCityTower input_shape:", input_shape, "\033[0m")

    def compute_output_shape(self, input_shape):
        return (input_shape['dep_icao'][0], self.output_dim)
    
        
    def process_input_and_embedding(self,inputs):
        dep_icao_emb = self.icao_embedding(inputs['dep_icao'])
        arr_icao_emb = self.icao_embedding(inputs['arr_icao'])
        dep_city_emb = self.city_embedding(inputs['dep_city_name'])
        arr_city_emb = self.city_embedding(inputs['arr_city_name'])
        dep_country_emb = self.country_embeding(inputs['dep_country_name'])
        arr_country_emb = self.country_embeding(inputs['arr_country_name'])
        dep_area_emb = self.area_embeding(inputs['dep_area_name'])
        arr_area_emb = self.area_embeding(inputs['arr_area_name'])

        x = tf.stack([
            dep_icao_emb,
            arr_icao_emb,
            dep_city_emb,
            arr_city_emb,
            dep_country_emb,
            arr_country_emb,
            dep_area_emb,
            arr_area_emb
        ], axis=1)

        return x

        
    def call(self, inputs, training=False):

        x=self.process_input_and_embedding(inputs)
       
        #注意力
        for attention_block in self.attention_blocks:
            x, _ = attention_block(x, training=training)

        x=self.flatten(x)
        x0=x

        
        res_out = x
        for residual_block in self.residual_blocks:
            res_out = residual_block(res_out, training=training)
            
        dcn_out = x
        for dcn_block in self.dcn_blocks:
            dcn_out = dcn_block(dcn_out, training=training)





        final_fusion = tf.concat([res_out, dcn_out], axis=-1)
    
        

        out = self.final_fusion(final_fusion)
        x0 = self.shourcut_fusion(x0)  ##x0残差链接

        out=out+x0
        # 调整输出维度为 output_dim
        out = self.output_dense(out)
        
        # 全局平均池化
        # out = tf.reduce_mean(out, axis=1)
        
        return out
    
    def get_config(self):
        config = super(CityTower, self).get_config()
        config.update({
            'icao_size': self.icao_size,
            'city_size': self.city_size,
            'country_size': self.country_size,
            'area_size': self.area_size,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'output_dim': self.output_dim,
            'units': self.units,
            'depth_mlp': self.depth_mlp,
            'depth_attention': self.depth_attention
        })
        
        return config

# 添加这个装饰器来注册模型
@register_keras_serializable()
class DualTowerModel(Model):
    def __init__(self, icao_size, city_size, country_size, area_size, supplier_size,aircraft_model_size,
                 embedding_dim=8, num_heads=2, output_dim=8, units=256,depth_mlp=2,depth_attention=2,**kwargs):
        super(DualTowerModel, self).__init__(**kwargs)
        
        self.icao_size = icao_size
        self.city_size = city_size
        self.country_size = country_size
        self.area_size = area_size
        self.supplier_size = supplier_size
        self.aircraft_model_size = aircraft_model_size


        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.temperature = 0.1
        self.units = units
        self.depth_mlp = depth_mlp
        self.depth_attention = depth_attention
        self.output_dim = output_dim
        
        self.supplier_tower = SupplierTower(
            icao_size=icao_size,
            city_size=city_size,
            supplier_size=supplier_size,
            aircraft_model_size=aircraft_model_size,
            country_size=country_size,
            area_size=area_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            output_dim=output_dim,
            units=units,
            depth_mlp=depth_mlp,
            depth_attention=depth_attention
        )
        
        self.city_tower = CityTower(
            icao_size=icao_size,
            city_size=city_size,
            country_size=country_size,
            area_size=area_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            output_dim=output_dim,
            units=units,
            depth_mlp=depth_mlp,
            depth_attention=depth_attention
        )

    def build(self, input_shape):
        # 创建一个输入来构建子层
        batch_size = 1
        dummy_inputs = {
            #城市塔
            'dep_icao': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_icao': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_city_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_city_name': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_area_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_area_name': tf.zeros((batch_size,), dtype=tf.int32),


            #航司塔输入
            'supplier_id': tf.zeros((batch_size,), dtype=tf.int32),
            'supplier_country_name': tf.zeros((batch_size,), dtype=tf.int32),
           
        }

        dummy_inputs['supplier_aircraft_scale_type'] = tf.zeros((batch_size,1), dtype=tf.int32)
        for i in range(1, 4):
            dummy_inputs[f'main_area_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)
    
        # 添加机型特征 (15个机型)
        for i in range(1, 16):
            dummy_inputs[f'aircraft_model_{i}'] = tf.zeros((batch_size,), dtype=tf.float32)

        for i in range(1, 7):
            dummy_inputs[f'base_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)

        for i in range(1, 7):
            dummy_inputs[f'base_city_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)
        
        # 构建两个塔，输入接受的形状是相同的，但是使用的部分是不同的
        self.supplier_tower(dummy_inputs)
        self.city_tower(dummy_inputs)
        
        super().build(input_shape)
        print("\033[93mDualTowerModel input_shape:", input_shape, "\033[0m")
      
        

    def compute_output_shape(self, input_shapes):
        batch_size = list(input_shapes.values())[0][0]
        return [(batch_size,)]
        
    def call(self, inputs, training=False):
        # 获取两个塔的输出
        supplier_vec = self.supplier_tower(inputs, training=training)
        city_vec = self.city_tower(inputs, training=training)
        
        # 确保维度正确性
        if len(supplier_vec.shape) == 3:
            supplier_vec = tf.reduce_mean(supplier_vec, axis=1)
        if len(city_vec.shape) == 3:
            city_vec = tf.reduce_mean(city_vec, axis=1)
            
        # L2归一化，并防止，分母范数为0
        supplier_vec = tf.nn.l2_normalize(supplier_vec,epsilon=1e-12, axis=-1)
        city_vec = tf.nn.l2_normalize(city_vec,epsilon=1e-12, axis=-1)
        
        # 计算点积得到相似度,值域为[-1,1],需要除以一个缩放因子
        similarity = tf.reduce_sum(supplier_vec * city_vec, axis=-1)
        # print(f"\n相似度:{similarity}")
        # print(f"similarity.shape:{similarity.shape}")

        similarity = similarity / self.temperature
        # print(f"\n相似度:{similarity}")
        # print(f"similarity.shape:{similarity.shape}")
        
        # 应用sigmoid激活函数
        output = tf.nn.sigmoid(similarity)
        # print(f"\noutput:{output}")
        # print(f"output.shape:{output.shape}")

      
       
        
        return output
    
    def get_config(self):
        config = super(DualTowerModel, self).get_config()
        config.update({
            'icao_size': self.icao_size,
            'city_size': self.city_size,
            'country_size': self.country_size,
            'area_size': self.area_size,
            'supplier_size': self.supplier_size,
            'aircraft_model_size': self.aircraft_model_size,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'output_dim': self.output_dim,
            'units': self.units,
            'depth_mlp': self.depth_mlp,
            'depth_attention': self.depth_attention
        })
        return config



def create_dual_tower_model(icao_size, city_size, country_size, area_size, supplier_size,aircraft_model_size, 
                            embedding_dim=8, num_heads=2, output_dim=8, units=256,depth_mlp=2,depth_attention=2):
    """
    创建双塔模型
    
    参数:
    city_vocab_size: 城市词汇表大小 = len(city_encoder.classes_)
    region_vocab_size: 地区词汇表大小 = len(region_encoder.classes_)
    supplier_vocab_size: 航司词汇表大小 = len(supplier_encoder.classes_)
    embedding_dim: embedding维度，默认8
    num_heads: 注意力头数，默认2
    """
    model = DualTowerModel(
        icao_size=icao_size,
        city_size=city_size,
        country_size=country_size,
        area_size=area_size,
        supplier_size=supplier_size,
        aircraft_model_size=aircraft_model_size,

        embedding_dim=embedding_dim,
        num_heads=num_heads,
        output_dim=output_dim,
        units=units,
        depth_mlp=depth_mlp,
        depth_attention=depth_attention
    )
    
     # 创建一个批次的示例输入来构建模型，结构相同，用于第一次初始化
    batch_size = 1
    dummy_inputs = {
            #城市塔
            'dep_icao': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_icao': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_city_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_city_name': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_country_name': tf.zeros((batch_size,), dtype=tf.int32),
            'dep_area_name': tf.zeros((batch_size,), dtype=tf.int32),
            'arr_area_name': tf.zeros((batch_size,), dtype=tf.int32),


            #航司塔输入
            'supplier_id': tf.zeros((batch_size,), dtype=tf.int32),
           
            'supplier_country_name': tf.zeros((batch_size,), dtype=tf.int32),
           
        }
    dummy_inputs['supplier_aircraft_scale_type'] = tf.zeros((batch_size,1), dtype=tf.int32)
    for i in range(1, 4):
        dummy_inputs[f'main_area_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)

    # 添加机型特征 (15个机型)
    for i in range(1, 16):
        dummy_inputs[f'aircraft_model_{i}'] = tf.zeros((batch_size,), dtype=tf.float32)

    for i in range(1, 7):
        dummy_inputs[f'base_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)

    for i in range(1, 7):
        dummy_inputs[f'base_city_{i}'] = tf.zeros((batch_size,), dtype=tf.int32)
    
    # 进行一次前向传播来构建模型
    _ = model(dummy_inputs)
    
    print("\033[33m" + f"dual_tower_model output_shape: {_.shape}" + "\033[0m")
    return model









def dual_tower_train(batch_size,epochs):
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


     # 加载预处理后的数据,再随机打乱
    df1=pd.read_csv(r"./data/data_encoded_part1.csv")
    df2=pd.read_csv(r"./data/data_encoded_part2.csv")
    df3=pd.read_csv(r"./data/data_encoded_part3.csv")
    df4=pd.read_csv(r"./data/data_encoded_part4.csv")
    df5=pd.read_csv(r"./data/data_encoded_part5.csv")

    encoded_df = pd.concat([df1,df2,df3,df4,df5],ignore_index=True) 
    encoded_df = encoded_df.sample(frac=1).reset_index(drop=True)

   

    ##抽取1000条数据小样本实验
    # encoded_df = encoded_df.head(1000)

    #显示label的分布
    # print(encoded_df['label'].value_counts())




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

    for i in range(1, 7):
        inputs[f'base_city_{i}'] = encoded_df[f'base_city_{i}'].values

    y = encoded_df['label'].values
    print("标签")
    print(y)
    print(len(y))


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
    def create_dataset(inputs, labels, batch_size, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(labels))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_dataset(train_inputs, train_y, batch_size, shuffle=False)
    val_dataset = create_dataset(val_inputs, val_y, batch_size, shuffle=False)
    test_dataset = create_dataset(test_inputs, test_y, batch_size, shuffle=False)

    del encoded_df  # 数据准备完成，删除df节省内存
    







    ##拿到各个特征的大小,准备模型构造
    icao_size = len(encoders['icao_encoder'].classes_)
    city_size = len(encoders['city_encoder'].classes_)
    country_size = len(encoders['country_encoder'].classes_)
    area_size = len(encoders['area_encoder'].classes_)
    supplier_size = len(encoders['supplier_encoder'].classes_)
    aircraft_model_size = len(encoders['aircraft_encoder'].classes_)

    ##模型超参数
    embedding_dim_input = 2
    num_heads_input = 2
    depth_mlp_input = 2
    depth_attention_input = 2
    units_input = 2
    output_dim = 2

    model = create_dual_tower_model(
        icao_size=icao_size,
        city_size=city_size,
        country_size=country_size,
        area_size=area_size,
        supplier_size=supplier_size,
        aircraft_model_size=aircraft_model_size,
        embedding_dim=embedding_dim_input,
        num_heads=num_heads_input,
        depth_mlp=depth_mlp_input,
        depth_attention=depth_attention_input,
        units=units_input,
        output_dim=output_dim
    )

    model.summary()


    # 3. 设置学习率调度
    initial_learning_rate = 0.001
    t_mul = 1.0                  # 每次重启后周期的倍数
    m_mul = 0.9                  # 每次重启后学习率最大值的倍数
    alpha = 0.000001                  # 最小学习率比例

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
    optimizer = Adam(learning_rate=lr_schedule)


    class TrainingMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            print(f"Training Loss: {logs['loss']:.4f}")
            print(f"Validation Loss: {logs['val_loss']:.4f}")
            print(f"Training Accuracy: {logs['binary_accuracy']:.4f}")
            print(f"Validation Accuracy: {logs['val_binary_accuracy']:.4f}")
    
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


  


    callbacks = [
        TrainingMonitor(),
    ]


 
    # 8. 训练模型
    history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    verbose=1,
    callbacks=callbacks
    )

    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
     # 9. 评估模型
    # 在测试集上进行预测
    test_predictions_prob = model.predict(test_dataset)
    test_predictions_binary = (test_predictions_prob > 0.5).astype(int)

    # 计算各种指标
    accuracy = accuracy_score(test_y, test_predictions_binary)
    precision = precision_score(test_y, test_predictions_binary)
    recall = recall_score(test_y, test_predictions_binary)
    f1 = f1_score(test_y, test_predictions_binary)

    

    # 训练后的损失和精度
    train_loss = history.history['loss']
    train_accuracy = history.history['binary_accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_binary_accuracy']


    print("训练和验证集：=====================================================================")

    # 打印所有训练过程中每个epoch的损失和精度
    for epoch in range(len(train_loss)):
        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss[epoch]:.4f}, Train Accuracy: {train_accuracy[epoch]:.4f}, "
              f"Val Loss: {val_loss[epoch]:.4f}, Val Accuracy: {val_accuracy[epoch]:.4f}")

    print("测试集：===========================================================================")

    # 计算混淆矩阵
    cm = confusion_matrix(test_y, test_predictions_binary)
    print("\n=== 混淆矩阵 ===")
    print(cm)

    # 计算AUC
    auc_score  = roc_auc_score(test_y, test_predictions_prob)
    print(f"\nAUC: {auc_score:.4f}")

    

    

    # 绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(test_y, test_predictions_prob)
    roc_auc = auc(fpr, tpr)
    print("\n thresholds : \n")
    print(thresholds)




    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
  

   

    print("\n=== 模型评估结果 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")


    # 保存完整模型
    model.save('./model/dual_tower_model_0123_2.keras')
    print("模型已保存为 ./model/dual_tower_model_0123_2.keras")
    print( f"embedding:{embedding_dim_input}")
    print(f"num_heads_input:{num_heads_input}")
    print(f"epochs: {epochs}")

    # 10. 绘制训练历史

    

    # 损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('model_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history.history['binary_accuracy'], label='train_accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='val_accuracy')
    plt.title('model_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    # plt.show()

    plt.savefig('./image_dual_tower_model_0123_1.png', format='png')  # 保存为 PNG 




if __name__ == "__main__":
    ###训练踩的大坑，坑壁坑啊，目前最佳参数：tmperature，设置为1，学习率为0.001
    dual_tower_train(epochs=2,batch_size=256)