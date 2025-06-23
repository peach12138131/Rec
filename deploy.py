import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import List, Dict, Tuple
from blocks import ResidualBlock, FeatureAttention, CrossLayer
from model import SupplierTower, CityTower, DualTowerModel

def prepare_icao_input(icao_list, icao_feature_path='./vectors/unique_icao_features.pkl'):
    """输入的icao数据为了方便可以是[[icao1,icao2],[icao3,icao4]],,每个子list前一个为dep_icao,后一个为arr_icao"""
    icao_df = pd.read_pickle(icao_feature_path)
    print(icao_df.columns)
    # Index(['orig_icao', 'orig_city_name', 'orig_country_name', 'orig_area_name',
    #    'icao', 'city_name', 'country_name', 'area_name'],
    #   dtype='object')
    
    inputs = {}  # 用于模型的输入,参考格式如下
    # inputs={
    #     #城市塔
    #     'dep_icao': 
    #     'arr_icao': 
    #     'dep_city_name': 
    #     'arr_city_name': 
    #     'dep_country_name': 
    #     'arr_country_name': 
    #     'dep_area_name': 
    #     'arr_area_name': 
    # }
    # 用for循环遍历icao_list
    # 初始化存储所有特征的列表
    dep_icaos, arr_icaos = [], []
    dep_cities, arr_cities = [], []
    dep_countries, arr_countries = [], []
    dep_areas, arr_areas = [], []
    
    # 遍历每对ICAO
    for dep_icao, arr_icao in icao_list:
        # 获取出发地特征
        dep_feature = icao_df[icao_df['orig_icao'] == dep_icao].iloc[0]  # 使用iloc[0]确保获取单行
        # 获取目的地特征
        arr_feature = icao_df[icao_df['orig_icao'] == arr_icao].iloc[0]
        
        # 添加到对应列表
        dep_icaos.append(dep_feature['icao'])
        arr_icaos.append(arr_feature['icao'])
        dep_cities.append(dep_feature['city_name'])
        arr_cities.append(arr_feature['city_name'])
        dep_countries.append(dep_feature['country_name'])
        arr_countries.append(arr_feature['country_name'])
        dep_areas.append(dep_feature['area_name'])
        arr_areas.append(arr_feature['area_name'])
    
    # 将所有特征组织成字典
    inputs = {
        'dep_icao': np.array(dep_icaos),
        'arr_icao': np.array(arr_icaos),
        'dep_city_name': np.array(dep_cities),
        'arr_city_name': np.array(arr_cities),
        'dep_country_name': np.array(dep_countries),
        'arr_country_name': np.array(arr_countries),
        'dep_area_name': np.array(dep_areas),
        'arr_area_name': np.array(arr_areas)
    }
    
    # 添加错误检查
    for key, value in inputs.items():
        if len(value) != len(icao_list):
            raise ValueError(f"特征 {key} 的长度 ({len(value)}) 与输入ICAO对的数量 ({len(icao_list)}) 不匹配")
    
    return inputs


def extract_icao_vectors(model, inputs, batch_size=32):
    """
    提取所有航司的特征向量，使用批次处理以避免内存溢出
    """
    from tqdm import tqdm
    
    with tf.device('/GPU:0'):
        # 获取样本数量
        num_samples = len(inputs['dep_icao'])
        
        # 创建输入tensor字典
        tensor_inputs = {
            key: tf.constant(value) for key, value in inputs.items()
        }
        
        # 初始化结果列表
        all_vectors = []
        
        # 计算总批次数
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        # 使用tqdm创建进度条
        with tqdm(total=total_batches, desc="Extracting ICAO vectors") as pbar:
            # 分批处理
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                
                # 为当前批次创建输入
                batch_inputs = {
                    key: tensor_inputs[key][i:end_idx] for key in tensor_inputs
                }
                
                # 获取当前批次的向量
                batch_vectors = model.city_tower(batch_inputs)
                
                if len(batch_vectors.shape) == 3:
                    batch_vectors = tf.reduce_mean(batch_vectors, axis=1)
                
                # L2归一化
                batch_vectors = tf.nn.l2_normalize(batch_vectors, axis=-1)
                
                # 添加到结果列表
                all_vectors.append(batch_vectors.numpy())
                
                # 更新进度条
                pbar.update(1)
        
        # 合并所有批次的结果
        icao_vectors = np.vstack(all_vectors)
        
        return icao_vectors


def load_supplier_vectors(vector_file: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    加载保存的供应商向量数据
    """
    with open(vector_file, 'rb') as f:
        save_data = pickle.load(f)
    
    vectors = np.array(list(save_data['vector_dict'].values()))
    ids = np.array(list(save_data['vector_dict'].keys()))
    vector_dict = save_data['vector_dict']
    
    return vectors, ids, vector_dict


def compute_similarity(icao_vectors: np.ndarray, supplier_vectors: np.ndarray, batch_size=32) -> np.ndarray:
    """
    计算 ICAO 向量和供应商向量之间的余弦相似度，使用批次处理
    
    Args:
        icao_vectors: shape (num_icao_pairs, vector_dim)
        supplier_vectors: shape (num_suppliers, vector_dim)
        batch_size: 批次大小
        
    Returns:
        similarity_matrix: shape (num_icao_pairs, num_suppliers)
    """
    num_icao_pairs = icao_vectors.shape[0]
    num_suppliers = supplier_vectors.shape[0]
    
    # 初始化结果矩阵
    similarity_matrix = np.zeros((num_icao_pairs, num_suppliers))
    
    # 分批计算
    for i in range(0, num_icao_pairs, batch_size):
        end_idx = min(i + batch_size, num_icao_pairs)
        
        # 计算当前批次的余弦相似度
        batch_similarity = tf.matmul(
            tf.constant(icao_vectors[i:end_idx]), 
            tf.constant(supplier_vectors), 
            transpose_b=True
        )
        
        # 保存到结果矩阵
        similarity_matrix[i:end_idx] = batch_similarity.numpy()
    
    return similarity_matrix


def get_top_n_recommendations(
    similarity_matrix: np.ndarray,
    supplier_ids: np.ndarray,
    icao_pairs: List[List[str]],
    supplier_df: pd.DataFrame,
    n: int = 5
) -> List[Dict]:
    """
    为每个 ICAO 对获取前 N 个推荐结果
    
    Args:
        similarity_matrix: 相似度矩阵 shape (num_icao_pairs, num_suppliers)
        supplier_ids: 供应商 ID 数组
        icao_pairs: 原始 ICAO 对列表
        n: 推荐数量
        
    Returns:
        recommendations: 包含每个 ICAO 对推荐结果的列表
    """
    recommendations_dict = {}
    
    for idx, (similarity_scores, icao_pair) in enumerate(zip(similarity_matrix, icao_pairs)):
        # 获取前 N 个最相似的索引
        top_n_indices = np.argsort(similarity_scores)[-n:][::-1]
        
        # 获取对应的供应商 ID 和相似度分数
        top_suppliers = supplier_ids[top_n_indices]
        top_scores = similarity_scores[top_n_indices]
        

        # 使用连字符连接ICAO对作为键
        icao_key = f"{icao_pair[0]}-{icao_pair[1]}"

        recommendations_dict[icao_key] = []
       
        
        for supplier_id, score in zip(top_suppliers, top_scores):
            # 初始化推荐信息字典
            recommendation_info = {
                'supplier_id': str(supplier_id),
                'similarity_score': float(score)
            }
            
            # 尝试查找供应商名称
            try:
                # 确保 supplier_id 是字符串类型进行比较
                supplier_id_str = str(supplier_id)
                filtered_df = supplier_df[supplier_df['orig_supplier_id'] == supplier_id_str]
                
                if not filtered_df.empty:
                    supplier_name = filtered_df['orig_supplier_name'].iloc[0]
                    recommendation_info['supplier_name'] = supplier_name
                else:
                    recommendation_info['supplier_name'] = "未知供应商"
            except Exception as e:
                # 出错时记录错误并设置默认名称
                print(f"查找供应商 {supplier_id} 时出错: {e}")
                recommendation_info['supplier_name'] = "未知供应商"
            
            # 添加到推荐列表
            recommendations_dict[icao_key].append(recommendation_info)
        
    return recommendations_dict

def inference(model, icao_list, supplier_vectors, supplier_df, BATCH_SIZE=32, n=5):
    inputs = prepare_icao_input(icao_list)
    print(inputs)

    # 使用批处理提取ICAO向量
    icao_vectors = extract_icao_vectors(model, inputs, batch_size=BATCH_SIZE)
    
    # 使用批处理计算相似度
    similarity_matrix = compute_similarity(icao_vectors, supplier_vectors, batch_size=BATCH_SIZE)
    
    # 获取推荐结果
    recommendations = get_top_n_recommendations(
        similarity_matrix,
        supplier_ids,
        icao_list,
        supplier_df=supplier_df,
        n=n
    )
    return recommendations


def save_recommendations_as_json(recommendations_dict, filename):
    """将推荐结果保存为JSON格式"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(recommendations_dict, f, ensure_ascii=False, indent=2)
    
    print(f"推荐结果已保存至 {filename}")

def load_recommendations_from_json(filename):
    """从JSON文件加载推荐结果"""
    with open(filename, 'r', encoding='utf-8') as f:
        recommendations_dict = json.load(f)
    
    return recommendations_dict

def search_recommendation(recommendations_dict, dep_icao, arr_icao):
    """在推荐结果字典中搜索特定ICAO对的推荐结果"""
    icao_key = f"{dep_icao}-{arr_icao}"
    return recommendations_dict.get(icao_key)




if __name__ == "__main__":
    # 设置批次大小
    BATCH_SIZE = 1024
    
    model_path = './model/contrastive_model_0225.keras'
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'DualTowerModel': DualTowerModel,
            'SupplierTower': SupplierTower,
            'CityTower': CityTower,
            'ResidualBlock': ResidualBlock,
            'FeatureAttention': FeatureAttention,
            'CrossLayer': CrossLayer
        },
        compile=False
    )
    # print(model)
    model.summary()
    
    supplier_vector_path = './vectors/supplier_vectors.pkl'
    supplier_vectors, supplier_ids, supplier_vector_dict = load_supplier_vectors(supplier_vector_path)
    
    supplier_df_path = './vectors/unique_supplier_features.pkl'
    supplier_df = pd.read_pickle(supplier_df_path)
    
    test_icao = [
        ['KTUS','KSAT'],
       

        
    ]
    
    # test_icao=[
    #     ['WSSL','VTBD'],
    #     ['OMDW','EGGW'],
    #     ['KLAX','KTEB'],
       
       
    # ]
    # df1=pd.read_csv("./data/combined_data_0214_clean.csv")
    # test_icao=df1[['dep_icao','arr_icao']].drop_duplicates().values.tolist()
    # print("开始计算所有结果")
    print("数据量",len(test_icao))
    recommendations = inference(model, test_icao, supplier_vectors, supplier_df, BATCH_SIZE=BATCH_SIZE, n=20)
    
    
    # print(recommendations)
  
            
    save_recommendations_as_json(recommendations, './rec_res/recommendations.json')
    rec_dict=load_recommendations_from_json('./rec_res/recommendations.json')

    # a=search_recommendation(rec_dict, 'LOWK', 'EGLF')
    # print(a)
  
    
