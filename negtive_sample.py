"""进行负样本采样"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed





sample_count={"random":0,"easy":0,"medium":0,"hard":0}



def route_sample(df, ratios, all_suppliers, sampling_types=None):
    if sampling_types is None:
        sampling_types = ['random', 'easy', 'medium', 'hard']
    
    # 1. 预处理优化：使用更高效的数据结构
    print("预处理航司区域信息...")
    suppliers_areas = dict(zip(
        all_suppliers['supplier_id'],
        all_suppliers['main_area_name'].str.split(',').map(set)
    ))
    
    # 2. 批处理优化：一次性获取所有unique routes
    max_ratio = max(ratios.values())
    
    # 预先选择需要的列，减少内存使用
    required_columns = ['dep_icao', 'arr_icao', 'dep_area_name', 'arr_area_name', 
                       'supplier_id'] + [col for col in df.columns if col not in [
                           'dep_icao', 'arr_icao', 'dep_area_name', 'arr_area_name', 
                           'supplier_id']]
    
    # 使用较小的数据集进行采样
    sampled_df = df[required_columns].sample(frac=max_ratio, random_state=42,replace=True)
    
    # 高效获取唯一路由
    print("获取唯一航线...")
    unique_routes = sampled_df[
        ['dep_icao', 'arr_icao', 'dep_area_name', 'arr_area_name','supplier_id']
    ].drop_duplicates()

    route_for_sample = sampled_df[
        ['dep_icao', 'arr_icao', 'dep_area_name', 'arr_area_name', 'supplier_id']
    ].drop_duplicates()
    
    # 3. 预处理航线信息（优化版本）
    print("预处理航线信息...")
    
    # 创建航线到供应商的映射
    print("创建航线-供应商映射...")
    route_supplier_map = df.groupby(['dep_icao', 'arr_icao'])['supplier_id'].agg(set).to_dict()
    
    # 创建模板映射
    print("创建模板映射...")
    template_map = {}
    # 使用更高效的迭代方式
   
    for idx, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0], desc="Processing rows"):
        key = (row['dep_icao'], row['arr_icao'])
        if key not in template_map:
            template_map[key] = row.to_dict()
    
    # 合并信息到route_info
    print("合并航线信息...")
    route_info = {}
    for _, row in tqdm(unique_routes.iterrows(), total=unique_routes.shape[0], desc="Processing unique routes"):
        route_info[(row['dep_icao'], row['arr_icao'])] = {
            'template': template_map.get((row['dep_icao'], row['arr_icao']), {}),
            'existing_suppliers': route_supplier_map.get((row['dep_icao'], row['arr_icao']), set())
        }
    
    # 4. 供应商筛选优化：预计算区域映射
    supplier_area_mapping = defaultdict(list)
    for sid, areas in suppliers_areas.items():
        for area in areas:
            supplier_area_mapping[area].append(sid)
    
    def filter_suppliers(route_info, route_existing, sampling_type):
        dep_area = route_info['dep_area_name']
        arr_area = route_info['arr_area_name']
        
        if sampling_type == 'random':
            return all_suppliers['supplier_id'].tolist()
        
        # 使用预计算的区域映射加速查找
        if sampling_type == 'easy':
            potential = set(all_suppliers['supplier_id']) - set(
                supplier_area_mapping[dep_area] + 
                supplier_area_mapping[arr_area]
            )
        elif sampling_type == 'medium':
            potential = set(
                supplier_area_mapping[dep_area] + 
                supplier_area_mapping[arr_area]
            )
        else:  # hard
            potential = set(supplier_area_mapping[dep_area]) & set(
                supplier_area_mapping[arr_area]
            )
            
        # print(f"sampling_type: {sampling_type}, potential suppliers: {len(potential)}")
        return list(potential - route_existing)
    
    def process_route(route, sampling_type, ratio_adjustment):
        try:
            route_key = (route.dep_icao, route.arr_icao)
            
                
            route_dict = {
                'dep_area_name': route.dep_area_name,
                'arr_area_name': route.arr_area_name
            }
            
            potential_suppliers = filter_suppliers(
                route_dict,
                route_info[route_key]['existing_suppliers'],
                sampling_type
            )
            
            if not potential_suppliers:
                
                return None
            
            if np.random.random() > ratio_adjustment:
                return None
            
            sample_count[sampling_type]+=1


            chosen_supplier_id = np.random.choice(potential_suppliers)
            chosen_supplier = all_suppliers[
                all_suppliers['supplier_id'] == chosen_supplier_id
            ].iloc[0]
            
            template = route_info[route_key]['template'].copy()
            supplier_columns = [
                'supplier_id', 'supplier_name', 'supplier_country_name',
                'pure_reg', 'aircraft_id', 'aircraft_model', 'main_area_name',
                'supplier_aircraft_scale_type', 'aircraft_models'
            ]
            
            for col in supplier_columns:
                template[col] = chosen_supplier[col]
            template['label'] = 0
            
            return pd.Series(template)
            
        except Exception as e:
            # print(f"处理航线时发生错误: {e}")
            return None
    
    # 5. 并行采样过程
    difficulty = {'random': '随机', 'easy': '简单', 'medium': '中等', 'hard': '困难'}
    neg_dfs = []
    
    for sampling_type in sampling_types:
        print(f"\n开始{difficulty[sampling_type]}负采样...")
        print(f"采样数量: {len(route_for_sample)}")
        ratio_adjustment = ratios[sampling_type]/max_ratio
        
        with ThreadPoolExecutor(max_workers=8) as executor:  # 增加工作线程数
            futures = []
            for route in tqdm(route_for_sample.itertuples(), desc=f"提交{difficulty[sampling_type]}任务进度"):
                future = executor.submit(process_route, route, sampling_type, ratio_adjustment)
                futures.append(future)
            
            neg_samples = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"处理{difficulty[sampling_type]}结果进度"):
                try:
                    result = future.result()
                    if result is not None:
                        neg_samples.append(result)
                except Exception as e:
                    print(f"处理结果时发生错误: {e}")
                    continue
        
        if neg_samples:
            neg_df_type = pd.DataFrame(neg_samples)
            print(f"{difficulty[sampling_type]}负采样完成!")
            print(f"生成负样本数量: {len(neg_df_type)}")
            print(f"正负样本比例: 1:{len(neg_df_type)/len(df):.2f}")
            neg_dfs.append(neg_df_type)
    
    return pd.concat(neg_dfs, ignore_index=True) if neg_dfs else pd.DataFrame()






def supplier_sample_hard(df, ratio):
    """
    构造困难负样本：
    对于某航司，统计历史航线，筛选历史航线中比例最高的区域，国家，甚至城市，
    但是没飞过的航线dep_icao-arr_icao作为负样本
    """
    # 首先对原始数据进行采样
    sampled_df = df.sample(frac=ratio, random_state=42)
    
    # 存储负样本
    neg_samples = []
    
    # 获取所有unique的航司
    unique_suppliers = sampled_df['supplier_id'].unique()
    
    print("开始生成困难负样本...")
    for supplier_id in tqdm(unique_suppliers, desc="处理航司进度"):
        # 获取该航司的所有历史航线
        supplier_routes = df[df['supplier_id'] == supplier_id]
        
        if len(supplier_routes) == 0:
            continue
            
        # 统计该航司的航线区域分布
        area_stats = {
            'dep': supplier_routes['dep_area_name'].value_counts(normalize=True),
            'arr': supplier_routes['arr_area_name'].value_counts(normalize=True),
            'dep_country': supplier_routes['dep_country_name'].value_counts(normalize=True),
            'arr_country': supplier_routes['arr_country_name'].value_counts(normalize=True),
            'dep_city': supplier_routes['dep_city_name'].value_counts(normalize=True),
            'arr_city': supplier_routes['arr_city_name'].value_counts(normalize=True)
        }
        
        # 获取该航司已运营的航线组合
        existing_routes = set(zip(supplier_routes['dep_icao'], supplier_routes['arr_icao']))
        
        # 获取该航司最常运营的区域和国家
        top_dep_area = area_stats['dep'].index[0]
        top_arr_area = area_stats['arr'].index[0]
        top_dep_country = area_stats['dep_country'].index[0]
        top_arr_country = area_stats['arr_country'].index[0]
        
        # 寻找符合条件但未运营的航线
        potential_routes = df[
            (
                # 起飞地在最常运营区域或国家
                ((df['dep_area_name'] == top_dep_area) |
                 (df['dep_country_name'] == top_dep_country)) &
                # 降落地在最常运营区域或国家
                ((df['arr_area_name'] == top_arr_area) |
                 (df['arr_country_name'] == top_arr_country))
            )
        ].drop_duplicates(['dep_icao', 'arr_icao'])
        
        # 过滤掉已经运营的航线
        for _, route in potential_routes.iterrows():
            route_key = (route['dep_icao'], route['arr_icao'])
            if route_key not in existing_routes:
                # 获取一个航线模板
                template = route.copy()
                
                # 更新航司信息
                supplier_info = supplier_routes.iloc[0]
                supplier_columns = [
                    'supplier_id', 'supplier_name', 'supplier_country_name',
                    'pure_reg', 'aircraft_id', 'aircraft_model', 'main_area_name',
                    'supplier_aircraft_scale_type', 'aircraft_models'
                ]
                template[supplier_columns] = supplier_info[supplier_columns]
                template['label'] = 0
                
                neg_samples.append(template)
                break  # 每个航司只取一个负样本
    
    
    neg_df = pd.DataFrame(neg_samples)
    
    print(f"\n航司困难负采样完成!")
    print(f"原始正样本数量: {len(df)}")
    print(f"处理的正样本数量: {len(sampled_df)}")
    print(f"生成负样本数量: {len(neg_df)}")
    print(f"正负样本比例: 1:{len(neg_df)/len(df):.2f}")
    print(f"生成的困难负样本分布:")
    print("\n起飞区域分布:")
    print(neg_df['dep_area_name'].value_counts(normalize=True).head())
    print("\n降落区域分布:")
    print(neg_df['arr_area_name'].value_counts(normalize=True).head())
    
    return neg_df







class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

def start(pos_df,save_path):
    print("开始处理数据...")
    # 7. 数据预处理优化
    df = pos_df[pos_df['main_area_name'].notna()].copy()
    print(f"原始数据量: {len(pos_df)}")
    print(f"过滤后数据量: {len(df)}")
    df['label'] = 1
    
    # 8. 内存优化：只保留必要列
    all_suppliers = df[
        ['supplier_id', 'supplier_name', 'supplier_country_name', 'pure_reg',
         'aircraft_id', 'aircraft_model', 'main_area_name',
         'supplier_aircraft_scale_type', 'aircraft_models']
    ].drop_duplicates()
    
    ratios = {'random': 0.5, 'easy': 0.7, 'medium': 1, 'hard': 1}
    # sampling_types = ['easy', 'medium', 'hard']
    # sampling_types = ['easy']
    sampling_types = ['easy','medium','hard']
    

    with Timer() as t:
        neg_df_route = route_sample(df, ratios, all_suppliers, sampling_types)
        # neg_df_supplier = supplier_sample_hard(df, 1)
        # neg_df = pd.concat([neg_df_route, neg_df_supplier], ignore_index=True)
        neg_df=neg_df_route
    
    print(f"总处理时间: {t.elapsed:.2f}秒")
    print(f"最终正负样本比例: 1:{len(neg_df)/len(df):.2f}")
    os.makedirs('./data', exist_ok=True)
    
    # contact_df = pd.concat([df, neg_df], ignore_index=True)  #先不拼接
    # contact_df.to_csv(save_path, index=False)

    neg_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    #data_slice_slice_1.csv data_slice_slice_2.csv  data_slice_slice_3.csv  data_slice_slice_4.csv  data_slice_slice_5.csv
    # pos_df=pd.read_csv("sampled_2000_records.csv")
   
   
    
    # save_path="./data/neg_sample.csv"

    # start(pos_df,save_path)
    # print(sample_count)
    

    pos_df=pd.read_csv("./data/data_slice_slice_1.csv")
    save_path="./data/neg_slice1.csv"
    start(pos_df,save_path)
    
    pos_df=pd.read_csv("./data/data_slice_slice_2.csv")
    save_path="./data/neg_slice2.csv"
    start(pos_df,save_path)

    pos_df=pd.read_csv("./data/data_slice_slice_3.csv")
    save_path="./data/neg_slice3.csv"
    start(pos_df,save_path)

    pos_df=pd.read_csv("./data/data_slice_slice_4.csv")
    save_path="./data/neg_slice4.csv"
    start(pos_df,save_path)

    pos_df=pd.read_csv("./data/data_slice_slice_5.csv")
    save_path="./data/neg_slice5.csv"
    start(pos_df,save_path)
   
