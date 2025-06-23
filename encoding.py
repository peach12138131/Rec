import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df1=pd.read_csv("./data/data_slice_slice_1.csv")
df2=pd.read_csv("./data/data_slice_slice_2.csv")
df3=pd.read_csv("./data/data_slice_slice_3.csv")
df4=pd.read_csv("./data/data_slice_slice_4.csv")
df5=pd.read_csv("./data/data_slice_slice_5.csv")

neg_df1=pd.read_csv("./data/neg_slice1.csv")
neg_df2=pd.read_csv("./data/neg_slice2.csv")
neg_df3=pd.read_csv("./data/neg_slice3.csv")
neg_df4=pd.read_csv("./data/neg_slice4.csv")
neg_df5=pd.read_csv("./data/neg_slice5.csv")





df = pd.concat([df1,df2,df3,df4,df5,neg_df1,neg_df2,neg_df3,neg_df4,neg_df5],ignore_index=True)


#先去除，暂时不需要的行，减小后续处理压力：dep_ts,arr_ts,supplier_name,pure_reg,aircraft_id,aircraft_model，dep_city_id,arr_city_id,dep_country_id,arr_country_id,dep_area_id,arr_area_id,update_time
df=df.drop(columns=['dep_ts','arr_ts','supplier_name','pure_reg','aircraft_id','aircraft_model','dep_city_id','arr_city_id','dep_country_id','arr_country_id','dep_area_id','arr_area_id','update_time'])


for col in df.columns:
    df[col] = df[col].astype(str)

def create_efficient_icao_city_dict(dataframe):
    """
    使用pandas的高效操作从大型DataFrame创建ICAO到城市的映射。
    """
    # 创建两个Series：一个用于出发，一个用于到达
    dep_series = dataframe.set_index('dep_icao')['dep_city_name']
    arr_series = dataframe.set_index('arr_icao')['arr_city_name']
    
    # 合并这两个Series，去除重复和NaN值
    combined = pd.concat([dep_series, arr_series]).dropna()
    # 保留每个ICAO的第一个出现的城市名称
    icao_city_dict = combined.drop_duplicates().to_dict()
    
    return icao_city_dict

print("创建ICAO到城市的映射")
icao_city_dict = create_efficient_icao_city_dict(df)



# 初始化编码器
print("初始化编码器")
icao_encoder = LabelEncoder()
city_encoder = LabelEncoder()  # 用于城市编码（起飞地和降落地共享）
country_encoder = LabelEncoder()  # 用于国家编码
area_encoder = LabelEncoder()  # 用于所有地区编码（包括运营区域）

supplier_encoder = LabelEncoder()  # 用于航司编码
aircraft_encoder = LabelEncoder()  # 用于机型编码




##对supplier_id进行编码
print("对supplier_id进行编码")
all_supplier_id=pd.concat([df['supplier_id']]).unique()
supplier_encoder.fit(all_supplier_id)
df['supplier_id'] = supplier_encoder.transform(df['supplier_id'])






#1 对icao 进行编码
print("对icao 进行编码")
df['aircraft_base'] = df['aircraft_base'].fillna('')
expanded_base = df['aircraft_base'].str.split(',', expand=True)
max_base = expanded_base.shape[1]  # 获取base数
expanded_base.columns = [f'base_{i+1}' for i in range(max_base)]
# 清理数据 - 处理空值和空格
for col in expanded_base.columns:
    expanded_base[col] = expanded_base[col].str.strip()
    expanded_base[col] = expanded_base[col].fillna('N')
    expanded_base[col] = expanded_base[col].replace(['', None], 'N')

# 将展开的列合并到原始数据框
df = pd.concat([df, expanded_base], axis=1)

#  删除原始的 aircraft_base 列
df = df.drop('aircraft_base', axis=1)


print("添加并映射base城市")
for i in range(max_base):
    base_col = f'base_{i+1}'
    city_col = f'base_city_{i+1}'
    
    # 创建一个函数，将ICAO映射到城市，如果找不到则为'N'
    def map_icao_to_city(icao):
        if icao == 'N':
            return 'N'
        return icao_city_dict.get(icao, 'N')
    
    # 在编码ICAO之前，先映射到城市
    df[city_col] = df[base_col].apply(map_icao_to_city)


all_icao=pd.concat([df['dep_icao'],
                     df['arr_icao']]
                     + [df[f'base_{i+1}'] for i in range(max_base)]).unique()

icao_encoder.fit(all_icao)
df['dep_icao'] = icao_encoder.transform(df['dep_icao'])
df['arr_icao'] = icao_encoder.transform(df['arr_icao'])
for i in range(max_base):
    col = f'base_{i+1}'
    df[col] = icao_encoder.transform(df[col])






#2 对城市进行编码
print("对城市进行编码")
base_city_cols = [f'base_city_{i+1}' for i in range(max_base)]
all_city = pd.concat([df['dep_city_name'], df['arr_city_name']] + [df[col] for col in base_city_cols]).unique()
city_encoder.fit(all_city)
df['dep_city_name'] = city_encoder.transform(df['dep_city_name'])
df['arr_city_name'] = city_encoder.transform(df['arr_city_name'])
for col in base_city_cols:
    df[col] = city_encoder.transform(df[col])


#3 对国家进行编码
print("对国家进行编码")
all_country=pd.concat([df['dep_country_name'],df['arr_country_name'],df["supplier_country_name"]]).unique()
country_encoder.fit(all_country)
df['dep_country_name'] = country_encoder.transform(df['dep_country_name'])
df['arr_country_name'] = country_encoder.transform(df['arr_country_name'])
df["supplier_country_name"] = country_encoder.transform(df["supplier_country_name"])



#4 对地区进行编码，
print("对地区进行编码")
#先处理航司运营区域

df['main_area_name'] = df['main_area_name'].fillna('')  # 先将空值填充为空字符串
expanded_areas = df['main_area_name'].str.split(',', expand=True)

# 3. 获取展开后的最大区域数量并重命名列
max_areas = expanded_areas.shape[1]  # 获取最大地区数
expanded_areas.columns = [f'main_area_{i+1}' for i in range(max_areas)]

# 清理数据 - 处理空值和空格
for col in expanded_areas.columns:
    expanded_areas[col] = expanded_areas[col].str.strip()
    expanded_areas[col] = expanded_areas[col].fillna('N')
    expanded_areas[col] = expanded_areas[col].replace(['', None], 'N')

# 将展开的列合并到原始数据框
df = pd.concat([df, expanded_areas], axis=1)



#  删除原始的 main_area_name 列
df = df.drop('main_area_name', axis=1)

# 获取所有需要编码的区域值
all_areas = pd.concat([
    df['dep_area_name'],
    df['arr_area_name']
] + [df[f'main_area_{i+1}'] for i in range(max_areas)]).unique()

# 对所有区域进行编码
area_encoder.fit(all_areas)

#  转换每个区域列
df['dep_area_name'] = area_encoder.transform(df['dep_area_name'])
df['arr_area_name'] = area_encoder.transform(df['arr_area_name'])
for i in range(max_areas):
    col = f'main_area_{i+1}'
    df[col] = area_encoder.transform(df[col])








#对航司拥有机型编码，最终只扩充15个列。对应扩列后，机型小于15个的数据，保留当前机型，空值填充为N。对应大于15个的数据，保留频率最高的15种机型
print("对航司拥有机型编码")
df['aircraft_models'] = df['aircraft_models'].fillna('')
# 展开所有机型并统计频率
all_models = df['aircraft_models'].str.split(',').explode().str.strip()
model_freq = all_models.value_counts()
print(model_freq)


all_aircraft_models = all_models.unique()
# 添加 'N' 到机型列表中
all_aircraft_models = np.append(all_aircraft_models, 'N')
all_aircraft_models = np.unique([x for x in all_aircraft_models if x not in ['', None]])

aircraft_encoder.fit(all_aircraft_models)

def process_aircraft_models(models_str):
    if pd.isna(models_str) or models_str == '' or models_str == 'N':
        return ['N'] * 15  # 空值填充15个N
    
    models = [m.strip() for m in models_str.split(',')]
    if len(models) <= 15:
        
        top_models =models
        return top_models + ['N'] * (15 - len(models))
    else:
    
        top_models=models
        
        return top_models[:15]



# expanded_aircraft_models = df['aircraft_models'].apply(process_aircraft_models)
# expanded_aircraft_models_df = pd.DataFrame(expanded_aircraft_models.tolist(), 
#                          columns=[f'aircraft_model_{i+1}' for i in range(15)],
#                          index=df.index)
from tqdm import tqdm

# 添加进度条
tqdm.pandas(desc="Processing aircraft models")
expanded_aircraft_models = df['aircraft_models'].progress_apply(process_aircraft_models)
expanded_aircraft_models_df = pd.DataFrame(expanded_aircraft_models.tolist(), 
                         columns=[f'aircraft_model_{i+1}' for i in range(15)],
                         index=df.index)

# 5. 合并回原始数据框
df = pd.concat([df, expanded_aircraft_models_df], axis=1)


# 6. 删除原始的 aircraft_models 列
df = df.drop('aircraft_models', axis=1)



for i in range(15):
    col = f'aircraft_model_{i+1}'
    df[col] = aircraft_encoder.transform(df[col])




# 4. 机队数目标准化
print("对机队数目标准化")
df['supplier_aircraft_scale_type'] = df['supplier_aircraft_scale_type'].astype(float)
fleet_size = df['supplier_aircraft_scale_type'].values
fleet_size_normalized = (fleet_size - np.mean(fleet_size)) / np.std(fleet_size)
#替换原始数据
df['supplier_aircraft_scale_type'] = fleet_size_normalized


##保存编码后的数据
print("保存编码后的数据")
# df.to_csv("./data/sample_data_encoded.csv",index=False)

# 替换代码最后的数据分割部分为以下内容：

# 按照dep_area_name和arr_area_name组合对数据进行分组
# 按照dep_area_name和arr_area_name组合对数据进行分组
print("按区域组合分组保存数据")

# 创建一个新列，用于区域组合
df['area_combo'] = df['dep_area_name'].astype(str) + '_' + df['arr_area_name'].astype(str)

# 获取所有唯一的组合
area_combinations = df['area_combo'].unique()
print(f"共有{len(area_combinations)}种区域组合")

# 创建从编码值映射回原始名称的字典
area_names = {i: name for i, name in enumerate(area_encoder.classes_)}

# 定义一个函数，用于获取可读的区域名称用于文件命名
def get_area_name(encoded_value):
    return area_names.get(encoded_value, f"未知区域_{encoded_value}")

# 设置阈值，小于该数量的组合将被归入"others"
THRESHOLD = 10000

# 创建一个DataFrame用于保存"others"类别的数据
others_df = pd.DataFrame(columns=df.columns)

# 创建一个字典，记录哪些组合被归入others
others_info = {}

# 按区域组合分割并保存数据
for combo in area_combinations:
    # 提取出发和到达区域代码
    dep_area_code, arr_area_code = map(int, combo.split('_'))
    
    # 获取原始区域名称
    dep_area_name = get_area_name(dep_area_code)
    arr_area_name = get_area_name(arr_area_code)
    
    # 为该组合筛选数据
    combo_df = df[df['area_combo'] == combo]
    
    # 检查数据量是否低于阈值
    if len(combo_df) < THRESHOLD:
        # 如果低于阈值，添加到others_df
        others_df = pd.concat([others_df, combo_df])
        # 记录信息
        others_info[f"{dep_area_name}_{arr_area_name}"] = len(combo_df)
        print(f"组合 {dep_area_name} 到 {arr_area_name} 的数据量为 {len(combo_df)}，低于阈值 {THRESHOLD}，将归入others")
    else:
        # 创建可读的文件名
        filename = f"./data/from_{dep_area_name}_{arr_area_name}_encoded.csv"
        
        # 保存数据
        combo_df = combo_df.drop('area_combo', axis=1)  # 删除临时列
        combo_df.to_csv(filename, index=False)
        print(f"保存 {dep_area_name} 到 {arr_area_name} 的数据，行数：{len(combo_df)}")

# 保存others数据（如果有的话）
if not others_df.empty:
    others_df = others_df.drop('area_combo', axis=1)  # 删除临时列
    others_df.to_csv("./data/others_encoded.csv", index=False)
    print(f"保存others数据，共 {len(others_df)} 行，包含 {len(others_info)} 种组合")
    
    # 保存others的详细信息到一个单独的CSV文件
    others_info_df = pd.DataFrame({
        '组合': list(others_info.keys()),
        '数据量': list(others_info.values())
    })
    others_info_df.to_csv("./data/others_info.csv", index=False)
    print("保存others详细信息到others_info.csv")

# 如果你计划进一步使用df，删除临时列
df = df.drop('area_combo', axis=1)

# 继续保存编码器
print("保存编码器")
import pickle
encoders = {
   'icao_encoder': icao_encoder,
   'city_encoder': city_encoder,
   'country_encoder': country_encoder,
   'area_encoder': area_encoder,
   'supplier_encoder': supplier_encoder,
   'aircraft_encoder': aircraft_encoder,
   'fleet_stats': {
        'mean': np.mean(fleet_size),
        'std': np.std(fleet_size)
    }
}
with open('./data/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)


icao_size = len(encoders['icao_encoder'].classes_)
city_size = len(encoders['city_encoder'].classes_)
country_size = len(encoders['country_encoder'].classes_)
area_size = len(encoders['area_encoder'].classes_)
supplier_size = len(encoders['supplier_encoder'].classes_)
aircraft_model_size = len(encoders['aircraft_encoder'].classes_)

print(f"icao_size: {icao_size}, city_size: {city_size}, country_size: {country_size}, area_size: {area_size}, supplier_size: {supplier_size}, aircraft_model_size: {aircraft_model_size}")
# 同时保存区域编码对照表，方便查阅
area_mapping_df = pd.DataFrame({
    '编码值': list(area_names.keys()),
    '区域名称': list(area_names.values())
})
area_mapping_df.to_csv('./data/区域编码映射.csv', index=False)
print("保存区域编码映射参考文件")












