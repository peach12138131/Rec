import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import json
import os
import time
from typing import List, Dict, Tuple
from tqdm import tqdm
from blocks import ResidualBlock, FeatureAttention, CrossLayer
from model import SupplierTower, CityTower, DualTowerModel

# 设置内存增长选项，避免一次性分配所有GPU内存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def prepare_icao_input(icao_list, icao_feature_path='./vectors/unique_icao_features.pkl'):
    """输入的icao数据为了方便可以是[[icao1,icao2],[icao3,icao4]]，每个子list前一个为dep_icao,后一个为arr_icao"""
    icao_df = pd.read_pickle(icao_feature_path)
    
    # 初始化存储所有特征的列表
    dep_icaos, arr_icaos = [], []
    dep_cities, arr_cities = [], []
    dep_countries, arr_countries = [], []
    dep_areas, arr_areas = [], []
    
    # 遍历每对ICAO
    for dep_icao, arr_icao in icao_list:
        # 获取出发地特征
        dep_feature = icao_df[icao_df['orig_icao'] == dep_icao].iloc[0]
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
    
    return inputs

def prepare_supplier_input(orig_supplier_id_list, supplier_feature_path='./vectors/unique_supplier_features.pkl'):
    """准备供应商特征输入"""
    supplier_df = pd.read_pickle(supplier_feature_path)
    
    # 初始化特征列表
    supplier_features = {
        'supplier_id': [],
        'supplier_country_name': [],
        'supplier_aircraft_scale_type': [],
    }
  
    # 添加main_area特征,提前准备字段
    for i in range(1, 4):
        supplier_features[f'main_area_{i}'] = []
        
    # 添加aircraft_model特征
    for i in range(1, 16):
        supplier_features[f'aircraft_model_{i}'] = []

    # 添加base特征
    for i in range(1, 7):
        supplier_features[f'base_{i}'] = []

    # 添加base_city特征
    for i in range(1, 7):
        supplier_features[f'base_city_{i}'] = []
    
    # 收集每个供应商的特征
    for orig_supplier_id in orig_supplier_id_list:
        supplier_row = supplier_df[supplier_df['orig_supplier_id'] == orig_supplier_id].iloc[0]
        
        supplier_features['supplier_id'].append(supplier_row['supplier_id'])
        supplier_features['supplier_country_name'].append(supplier_row['supplier_country_name'])
        supplier_features['supplier_aircraft_scale_type'].append(supplier_row['supplier_aircraft_scale_type'])
        
        for i in range(1, 4):
            supplier_features[f'main_area_{i}'].append(supplier_row[f'main_area_{i}'])
            
        for i in range(1, 16):
            supplier_features[f'aircraft_model_{i}'].append(supplier_row[f'aircraft_model_{i}'])

        for i in range(1, 7):
            supplier_features[f'base_{i}'].append(supplier_row[f'base_{i}'])

        for i in range(1, 7):
            supplier_features[f'base_city_{i}'].append(supplier_row[f'base_city_{i}'])
    
    # 转换为numpy数组
    return {k: np.array(v) for k, v in supplier_features.items()}

def prepare_batch_inputs(icao_inputs: Dict, supplier_inputs: Dict) -> Dict:
    """
    准备批量计算的输入
    将每个ICAO对与每个供应商组合，形成完整的输入矩阵
    """
    n_routes = len(icao_inputs['dep_icao'])
    n_suppliers = len(supplier_inputs['supplier_id'])
    
    # 扩展ICAO特征以匹配供应商数量
    expanded_icao = {}
    for key, value in icao_inputs.items():
        # 对每个路线重复供应商次数
        expanded = np.repeat(value, n_suppliers, axis=0)
        expanded_icao[key] = expanded
    
    # 扩展供应商特征以匹配路线数量
    expanded_supplier = {}
    for key, value in supplier_inputs.items():
        # 对每个供应商特征在每个路线上重复
        expanded = np.tile(value, n_routes)
        expanded_supplier[key] = expanded
    
    # 合并所有特征
    combined_inputs = {**expanded_icao, **expanded_supplier}
    
    return combined_inputs



def get_top_n_recommendations_batched(
    model,
    icao_list: List[List[str]],
    supplier_list: List[str],
    supplier_df: pd.DataFrame,
    n: int = 5,
    route_batch_size: int = 100,  # 每批处理的航线数
    supplier_feature_path: str = './vectors/unique_supplier_features.pkl',
    output_path: str = './rec_res/all_recommendations.json',
    checkpoint_frequency: int = 10  # 每处理多少批次保存一次
) -> List[List[Dict]]:
    """
    使用批处理的方式为每个ICAO对获取前N个推荐结果，并定期保存结果
    
    Args:
        model: 训练好的模型
        icao_list: ICAO机场对列表
        supplier_list: 供应商ID列表
        supplier_df: 包含供应商信息的DataFrame
        n: 返回前n个推荐结果
        route_batch_size: 每批处理的航线数
        supplier_feature_path: 供应商特征文件路径
        output_path: 推荐结果输出路径
        checkpoint_frequency: 每处理多少批次保存一次
    
    Returns:
        recommendations: 每个ICAO对的前N个推荐供应商列表
    """
    start_time = time.time()
    total_routes = len(icao_list)
    
    print(f"总航线数: {total_routes}, 总供应商数: {len(supplier_list)}")
    
    # 预先准备所有供应商特征（只需准备一次）
    print("准备供应商特征...")
    supplier_inputs = prepare_supplier_input(supplier_list, supplier_feature_path)
    
    # 初始化结果存储 - 使用字典，键为route_key
    all_recommendations = {}
    
    # 检查是否存在之前的结果文件
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                all_recommendations = json.load(f)
            print(f"已加载现有结果文件，包含{len(all_recommendations)}条记录")
        except json.JSONDecodeError:
            print(f"现有文件不是有效的JSON格式，将创建新文件")
    
    # 对航线进行分批
    route_batches = [
        icao_list[i:i + route_batch_size] 
        for i in range(0, total_routes, route_batch_size)
    ]
    
    print(f"开始分批处理，共{len(route_batches)}批...")
    
    # 跟踪已处理的航线数
    processed_routes = 0
    
    # 分批处理航线
    with tf.device('/GPU:0'):
        for batch_idx, route_batch in enumerate(tqdm(route_batches, desc="处理航线批次")):
            # 过滤掉已经处理过的航线
            filtered_batch = []
            filtered_indices = []
            
            for i, (dep_icao, arr_icao) in enumerate(route_batch):
                route_key = f"{dep_icao}-{arr_icao}"
                if route_key not in all_recommendations:
                    filtered_batch.append([dep_icao, arr_icao])
                    filtered_indices.append(i)
            
            # 如果该批次中所有航线都已处理，则跳过
            if not filtered_batch:
                continue
                
            # 准备该批次的ICAO输入
            icao_inputs = prepare_icao_input(filtered_batch)
            
            # 准备批量计算的输入
            batch_inputs = prepare_batch_inputs(icao_inputs, supplier_inputs)
            
            # 转换为TensorFlow张量
            tensor_inputs = {k: tf.constant(v) for k, v in batch_inputs.items()}
            
            # 使用GPU进行预测
            predictions = model(tensor_inputs)
            
            # 将预测结果重塑为(batch_size, n_suppliers)形状
            batch_predictions = predictions.numpy().reshape(len(filtered_batch), -1)
            
            # 为每个航线获取前N个推荐
            for idx, route_predictions in enumerate(batch_predictions):
                dep_icao, arr_icao = filtered_batch[idx]
                route_key = f"{dep_icao}-{arr_icao}"
                
                # 获取前N个最高分的索引
                top_n_indices = np.argsort(route_predictions)[-n:][::-1]
                
                # 获取对应的供应商信息
                route_recommendations = []
                for i in top_n_indices:
                    orig_supplier_id = supplier_list[i]
                    supplier_info = supplier_df[supplier_df['orig_supplier_id'] == orig_supplier_id].iloc[0]
                    
                    recommendation = {
                        'orig_supplier_id': orig_supplier_id,
                        'orig_supplier_name': supplier_info['orig_supplier_name'],
                        'orig_country_name': supplier_info['orig_country_name'],
                        'score': float(route_predictions[i])
                    }
                    route_recommendations.append(recommendation)
                
                # 添加到结果字典
                all_recommendations[route_key] = route_recommendations
                processed_routes += 1


            
            # 每处理checkpoint_frequency批次就保存一次结果
            if (batch_idx + 1) % checkpoint_frequency == 0:
                # 按ICAO首字母分组
                grouped_recommendations = {}
                for route_key, recs in all_recommendations.items():
                    # 提取dep_icao
                    dep_icao = route_key.split('-')[0]
                    first_letter = dep_icao[0]
                    
                    if first_letter not in grouped_recommendations:
                        grouped_recommendations[first_letter] = {}
                    
                    grouped_recommendations[first_letter][route_key] = recs

                # 确保输出目录存在
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # 保存各个分组文件
                for letter, group_data in grouped_recommendations.items():
                    group_output_path = os.path.join(output_dir, f"{letter}_recommendations.json")
                    with open(group_output_path, 'w', encoding='utf-8') as f:
                        json.dump(group_data, f, ensure_ascii=False, indent=2)
                
                # 打印保存信息
                total_routes = sum(len(group) for group in grouped_recommendations.values())
                print(f"\n已将{total_routes}条推荐结果分组保存 (批次{batch_idx+1}/{len(route_batches)})")
                print(f"分组文件: {', '.join([f'{letter}({len(data)}条)' for letter, data in grouped_recommendations.items()])}")


    
    # 处理完所有批次后，再次保存结果以确保所有数据都被保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_recommendations, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    print(f"处理完成! 总耗时: {total_time:.2f} 秒")
    print(f"平均每个航线处理时间: {total_time / processed_routes:.6f} 秒" if processed_routes > 0 else "没有需要处理的新航线")
    print(f"推荐结果已保存至 {output_path}，共{len(all_recommendations)}条记录")
    
    
    
    return all_recommendations

if __name__ == "__main__":
    # 加载模型
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
    
    # 加载供应商向量
    supplier_vector_path = './vectors/unique_supplier_features.pkl'
    supplier_df = pd.read_pickle(supplier_vector_path)

    # 暂时以所有supplier_id作为候选
    orig_supplier_id_list = supplier_df['orig_supplier_id'].unique()
    
    # # 测试数据
    test_icao = [
        ['LOWK', 'EGLF'],
        ['KSLE', 'PANC'],
        ['KMCC', 'KNUQ'],
    ]

    df1=pd.read_csv("./data/all_data_20250226.csv")
    test_icao=df1[['dep_icao','arr_icao']].drop_duplicates().values.tolist()
    print("开始计算所有结果")
    print("数据量",len(test_icao))
    # 使用批处理方式获取推荐结果
    recommendations = get_top_n_recommendations_batched(
        model, 
        test_icao, 
        orig_supplier_id_list, 
        supplier_df, 
        output_path = './rec_res/all_recommendations.json',
        n=20,
        route_batch_size=1, # 可以根据GPU内存调整,换算逻辑batch_size=route_batch_size*航司数量
        checkpoint_frequency=1  ## 每处理多少批次保存一次
    )
    print(recommendations)
    
    # # 打印结果
    # for i, route_recommendations in enumerate(recommendations):
    #     print(f"\nRoute {test_icao[i][0]} -> {test_icao[i][1]} recommendations:")
    #     for j, rec in enumerate(route_recommendations, 1):
    #         print(f"{j}. Supplier: {rec['orig_supplier_name']} (ID: {rec['orig_supplier_id']}, Country: {rec['orig_country_name']}, Score: {rec['score']:.4f})")
    
    # save_recommendations_to_json(test_icao, recommendations, './rec_res/route_recommendations.json')