
import pandas as pd
import numpy as np
import pickle
import os
if not os.path.exists('./vectors'):
    os.makedirs('./vectors')
import json
import glob



##加载未被编码的正负样本集

p_df1=pd.read_csv("./data/data_slice_slice_1.csv")
p_df2=pd.read_csv("./data/data_slice_slice_2.csv")
p_df3=pd.read_csv("./data/data_slice_slice_3.csv")
p_df4=pd.read_csv("./data/data_slice_slice_4.csv")
p_df5=pd.read_csv("./data/data_slice_slice_5.csv")
pos_df=pd.concat([p_df1,p_df2,p_df3,p_df4,p_df5],ignore_index=True)
for col in pos_df.columns:
    pos_df[col] = pos_df[col].astype(str)

##加载编码样本
all_files = glob.glob("./data/*encoded.csv")
    
# 读取并合并所有文件
df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

# 合并所有数据框
encoded_df = pd.concat(df_list, ignore_index=True)
print(f"成功加载并合并所有数据")


def create_supplier_name_dict(pos_df):
    """
    从正样本数据创建供应商ID与名称的映射字典
    
    参数:
   
    返回:
    supplier_dict: supplier_id到supplier_name的映射字典
    """
   
    
    # 检查必要的列是否存在
    if 'supplier_id' not in pos_df.columns or 'supplier_name' not in pos_df.columns:
        raise ValueError("输入CSV文件必须包含'supplier_id'和'supplier_name'列")
    
    # 提取供应商ID和名称，并去重
    supplier_df = pos_df[['supplier_id', 'supplier_name']].drop_duplicates()
    print(f"找到 {len(supplier_df)} 个唯一的供应商记录")
    
    # 创建映射字典
    supplier_dict = dict(zip(supplier_df['supplier_id'], supplier_df['supplier_name']))
    
    # 创建反向映射（名称到ID）
    supplier_name_to_id = dict(zip(supplier_df['supplier_name'], supplier_df['supplier_id']))
    
    # 确保vectors目录存在
    os.makedirs('./vectors', exist_ok=True)
    
     # 保存字典为JSON文件
    json_path = './vectors/supplier_id_name_map.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(supplier_dict, f, ensure_ascii=False, indent=4)
        
    # 保存反向映射
    reverse_json_path = './vectors/supplier_name_id_map.json'
    with open(reverse_json_path, 'w', encoding='utf-8') as f:
        json.dump(supplier_name_to_id, f, ensure_ascii=False, indent=4)
    
    # 保存为pickle文件以便快速加载
    pkl_path = './vectors/supplier_id_name_map.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(supplier_dict, f)
        
    # 保存反向映射的pickle文件
    reverse_pkl_path = './vectors/supplier_name_id_map.pkl'
    with open(reverse_pkl_path, 'wb') as f:
        pickle.dump(supplier_name_to_id, f)
    
    # 验证保存的文件
    print(f"文件已保存：")
    print(f"1. ID到名称映射JSON文件: {json_path}")
    print(f"2. 名称到ID映射JSON文件: {reverse_json_path}")
    print(f"3. ID到名称映射PKL文件: {pkl_path}")
    print(f"4. 名称到ID映射PKL文件: {reverse_pkl_path}")
    
    # 打印一些样本以便检查
    print("\n供应商ID到名称映射示例:")
    sample_dict = {k: supplier_dict[k] for k in list(supplier_dict.keys())[:5]} if supplier_dict else {}
    for supplier_id, name in sample_dict.items():
        print(f"  ID {supplier_id} -> {name}")
    
   


def get_icao_feature(encoded_df):
    """保存两份文件,pkl是机器读取,json是人工方便读取检查,都保留文字特征,和encoding特征"""

    with open('./data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    icao_encoder = encoders['icao_encoder']
    city_encoder = encoders['city_encoder']  # 用于城市编码（起飞地和降落地共享）
    country_encoder = encoders['country_encoder']  # 用于国家编码
    area_encoder = encoders['area_encoder'] 

    dep_features = encoded_df[['dep_icao', 'dep_city_name', 'dep_country_name', 'dep_area_name', 
                              'dep_latitude', 'dep_longitude']].drop_duplicates(subset=['dep_icao'])
    
    arr_features = encoded_df[['arr_icao', 'arr_city_name', 'arr_country_name', 'arr_area_name',
                              'arr_latitude', 'arr_longitude']].drop_duplicates(subset=['arr_icao'])
    
     # 重命名列
    dep_features.columns = ['icao', 'city_name', 'country_name', 'area_name', 'latitude', 'longitude']
    arr_features.columns = ['icao', 'city_name', 'country_name', 'area_name', 'latitude', 'longitude']
    
    # 合并，并确保ICAO代码唯一
    all_features = pd.concat([dep_features, arr_features]).drop_duplicates(subset=['icao'])

    # 创建反向映射字典 (一次性批量转换，提高效率)
    icao_reverse_map = {i: v for i, v in enumerate(icao_encoder.classes_)}
    city_reverse_map = {i: v for i, v in enumerate(city_encoder.classes_)}
    country_reverse_map = {i: v for i, v in enumerate(country_encoder.classes_)}
    area_reverse_map = {i: v for i, v in enumerate(area_encoder.classes_)}


     # 添加原始名称列
    all_features['orig_icao'] = all_features['icao'].map(lambda x: icao_reverse_map.get(x, str(x)))
    all_features['orig_city_name'] = all_features['city_name'].map(lambda x: city_reverse_map.get(x, str(x)))
    all_features['orig_country_name'] = all_features['country_name'].map(lambda x: country_reverse_map.get(x, str(x)))
    all_features['orig_area_name'] = all_features['area_name'].map(lambda x: area_reverse_map.get(x, str(x)))
    
    






    # 保存到pkl文件
    pkl_path = './vectors/unique_icao_features.pkl'
    all_features.to_pickle(pkl_path)

    # 保存到json文件,保存两份文件,pkl是机器读取,json是人工方便读取检查,都保留文字特征,和encoding特征
    json_data = all_features.set_index('orig_icao').to_dict('index')
    json_path = './vectors/unique_icao_features.json'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    # 验证保存的文件
    print(f"文件已保存：")
    print(f"1. PKL 文件: {pkl_path}")
    print(f"2. JSON 文件: {json_path}")

    # 查看结果
    print("处理后的唯一 ICAO 特征表:")
    print(all_features.head())
    print("\n特征表的形状:", all_features.shape)

    loaded_df = pd.read_pickle(pkl_path)
    print("\nPKL 文件读取测试 (前3行):")
    print(loaded_df.head(3))

    # 读取 json
    with open(json_path, 'r', encoding='utf-8') as f:
        loaded_json = json.load(f)
    print("\nJSON 文件读取测试 (第一个记录):")
    print(list(loaded_json.items())[0])








def get_supplier_feature(encoded_df):

    with open('./data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
        country_encoder = encoders['country_encoder']  # 用于国家编码
        area_encoder = encoders['area_encoder'] 
        supplier_encoder = encoders['supplier_encoder']
        aircraft_encoder = encoders['aircraft_encoder']
        fleet_size_encoder = encoders['fleet_stats']
        icao_encoder = encoders['icao_encoder']    
        city_encoder = encoders['city_encoder']

    supplier_reverse_map = {i: v for i, v in enumerate(supplier_encoder.classes_)} if supplier_encoder else {}
    country_reverse_map = {i: v for i, v in enumerate(country_encoder.classes_)} if country_encoder else {}
    area_reverse_map = {i: v for i, v in enumerate(area_encoder.classes_)} if area_encoder else {}
    aircraft_reverse_map = {i: v for i, v in enumerate(aircraft_encoder.classes_)} if aircraft_encoder else {}
    base_reverse_map = {i: v for i, v in enumerate(icao_encoder.classes_)} if icao_encoder else {}
    city_reverse_map = {i: v for i, v in enumerate(city_encoder.classes_)} if city_encoder else {}
    print("检查city_reverse_map的内容:")
    for k, v in list(city_reverse_map.items())[:5]:  # 只打印前5个
        print(f"Key: {k}, Value: {v}, Type: {type(v)}")

    ##base_1 -- base_3 是航司基地, main_area_1 -- main_area_3 是航司运营区域, aircraft_model_1  -- aircraft_model_15 是航司拥有机型
    base_columns = [col for col in encoded_df.columns if col.startswith('base_') and not col.startswith('base_city_')]
    area_columns = [col for col in encoded_df.columns if col.startswith('main_area_')]
    aircraft_columns = [col for col in encoded_df.columns if col.startswith('aircraft_model_')]
    city_columns = [col for col in encoded_df.columns if col.startswith('base_city_')]
    columns_to_extract = ['supplier_id', 'supplier_country_name', 'supplier_aircraft_scale_type'] + base_columns + area_columns + aircraft_columns + city_columns
    # 提取所有航空公司基本信息
    # 筛选出存在于DataFrame中的列
   
    # 一次性提取所有列，然后按供应商ID去重
    supplier_df = encoded_df[columns_to_extract].drop_duplicates(subset=['supplier_id'], keep='first').reset_index(drop=True)

    

    print(supplier_df.head())
    print("DataFrame的所有列名:")
    print(supplier_df.columns.tolist())

   

    print("对航司id进行编码")
    supplier_df['orig_supplier_id'] = supplier_df['supplier_id'].map(lambda x: supplier_reverse_map.get(x, str(x)))

     # 加载供应商ID到名称的映射字典
   
    with open('./vectors/supplier_id_name_map.pkl', 'rb') as f:
        supplier_name_dict = pickle.load(f)
    print(f"加载了 {len(supplier_name_dict)} 个供应商ID到名称的映射")
    
    # 添加供应商名称列
    supplier_df['orig_supplier_name'] = supplier_df['orig_supplier_id'].map(
        lambda x: supplier_name_dict.get(str(x), supplier_name_dict.get(int(x) if str(x).isdigit() else x, "unknown")))
   

    print("对航司国家进行编码")
    supplier_df['orig_country_name'] = supplier_df['supplier_country_name'].map(lambda x: country_reverse_map.get(x, str(x)) if pd.notna(x) else None)

     # 为基地添加原始名称
    print("为基地添加原始名称")
    for col in base_columns:
        if col in supplier_df.columns:
            orig_col = f'orig_{col}'
            supplier_df[orig_col] = supplier_df[col].map(
                lambda x: base_reverse_map.get(x, str(x)) if pd.notna(x) else None)
    
    # 为运营区域添加原始名称
    print("为运营区域添加原始名称")
    for col in area_columns:
        if col in supplier_df.columns:
            orig_col = f'orig_{col}'
            supplier_df[orig_col] = supplier_df[col].map(
                lambda x: area_reverse_map.get(x, str(x)) if pd.notna(x) else None)
    
    # 为机型添加原始名称
    print("为机型添加原始名称")
    for col in aircraft_columns:
        if col in supplier_df.columns:
            orig_col = f'orig_{col}'
            supplier_df[orig_col] = supplier_df[col].map(
                lambda x: aircraft_reverse_map.get(x, str(x)) if pd.notna(x) else None)
            


    
    # 为基地城市添加原始名称
    print("为基地城市添加原始名称")
    for col in city_columns:
        if col in supplier_df.columns:
            orig_col = f'orig_{col}'
            supplier_df[orig_col] = supplier_df[col].map(
                lambda x: city_reverse_map.get(x, str(x)) if pd.notna(x) else None)
            


    pkl_path = './vectors/unique_supplier_features.pkl'
    supplier_df.to_pickle(pkl_path)

    # 保存到json文件,保存两份文件,pkl是机器读取,json是人工方便读取检查,都保留文字特征,和encoding特征
    json_data = supplier_df.set_index('orig_supplier_id').to_dict('index')
    json_path = './vectors/unique_supplier_features.json'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    # 验证保存的文件
    print(f"文件已保存：")
    print(f"1. PKL 文件: {pkl_path}")
    print(f"2. JSON 文件: {json_path}")

    # 查看结果
    print("处理后的唯一 ICAO 特征表:")
    print(supplier_df.head())
    print("\n特征表的形状:", supplier_df.shape)


    loaded_df = pd.read_pickle(pkl_path)
    print("\nPKL 文件读取测试 (前3行):")
    print(loaded_df.head(3))

    # 读取 json
    with open(json_path, 'r', encoding='utf-8') as f:
        loaded_json = json.load(f)
    print("\nJSON 文件读取测试 (第一个记录):")
    print(list(loaded_json.items())[0])




import tensorflow as tf
from blocks import ResidualBlock, FeatureAttention, CrossLayer
from model import SupplierTower,CityTower,DualTowerModel


def extract_supplier_vectors(model, inputs,  suppliers_df):
    """
    提取所有航司的特征向量
    """
    with tf.device('/GPU:0'):
        tensor_inputs = {
            key: tf.constant(value) 
            for key, value in inputs.items()
        }
        
        # 获取向量
        supplier_vectors = model.supplier_tower(tensor_inputs)
        if len(supplier_vectors.shape) == 3:
            supplier_vectors = tf.reduce_mean(supplier_vectors, axis=1)
        supplier_vectors = tf.nn.l2_normalize(supplier_vectors, axis=-1)
        supplier_vectors = supplier_vectors.numpy()
        
        # 直接用 suppliers_df 中的 orig_supplier_id 创建字典
        vector_dict = dict(zip(suppliers_df['orig_supplier_id'], supplier_vectors))
        
        # 创建用于相似度计算的格式
        save_data = {
            'vectors': np.array(list(vector_dict.values())),  # 向量矩阵
            'ids': np.array(list(vector_dict.keys())),       # 对应的原始ID列表
            'vector_dict': vector_dict                        # 原始ID到向量的映射字典
        }
        
        return save_data



def get_supplier_vectors(model_path='./model/contrastive_model_0217.keras'):

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'DualTowerModel': DualTowerModel,
            'supplierTower': SupplierTower,
            'CityTower': CityTower,
            'ResidualBlock': ResidualBlock,
            'FeatureAttention': FeatureAttention,
            'CrossLayer': CrossLayer
        },
        compile=False
    )
    print(model)
    model.summary()

    suppliers_df = pd.read_pickle('./vectors/unique_supplier_features.pkl')
    print(suppliers_df.columns)

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
    
    for i in range(1, 7):
        inputs[f'base_city_{i}'] = suppliers_df[f'base_city_{i}'].values

    save_data = extract_supplier_vectors(model, inputs,suppliers_df)

    # 保存航司向量
    with open('./vectors/supplier_vectors.pkl', 'wb') as f:
        pickle.dump(save_data, f)


    






    

    


    
    
  
    






if __name__ == "__main__":
    
    get_icao_feature(encoded_df=encoded_df)

    create_supplier_name_dict(pos_df=pos_df)
    get_supplier_feature(encoded_df=encoded_df)

    get_supplier_vectors('./model/contrastive_model_0225.keras')

    






    

    


    
    
  
    





