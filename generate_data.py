import random
import csv
from datetime import datetime, timedelta

# 配置航司和其他常量
airlines = ['航司A', '航司B', '航司C', '航司D', '航司E', '航司F', '航司G', '航司H', '航司I', '航司J']
regions = ['华东', '华南', '华北', '西南', '东北', '西北', '中南', '东南']
aircraft_fleet = {  # 模拟不同航司的机队数量
    '航司A': 150, '航司B': 120, '航司C': 80, '航司D': 200, '航司E': 180,
    '航司F': 220, '航司G': 170, '航司H': 90, '航司I': 110, '航司J': 160
}
airport_pairs = [
    ('上海', '北京'), ('广州', '深圳'), ('成都', '重庆'), ('杭州', '南京'),
    ('北京', '天津'), ('西安', '兰州'), ('武汉', '长沙'), ('沈阳', '大连'),
    ('香港', '澳门'), ('厦门', '福州'), ('青岛', '济南'), ('郑州', '西安'),
    ('昆明', '贵阳'), ('乌鲁木齐', '兰州'), ('呼和浩特', '包头'), ('西宁', '银川'),
    ('拉萨', '日喀则'), ('海口', '三亚'), ('南宁', '桂林'), ('南昌', '九江'),
    ('合肥', '芜湖'), ('石家庄', '保定'), ('太原', '大同'), ('哈尔滨', '齐齐哈尔'),
    ('长春', '吉林'), ('呼伦贝尔', '满洲里'), ('锡林郭勒', '乌兰察布'), ('阿拉善', '巴彦淖尔'),
    ('克拉玛依', '塔城'), ('阿勒泰', '喀纳斯'), ('和田', '喀什'), ('阿克苏', '库尔勒'),
    ('哈密', '吐鲁番'), ('昌吉', '博尔塔拉'), ('塔城', '阿勒泰'), ('阿勒泰', '塔城')
]

# 随机生成日期时间
def generate_random_datetime():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_time = start_date + timedelta(days=random_days, hours=random.randint(0, 23), minutes=random.randint(0, 59))
    return random_time.strftime('%Y-%m-%d %H:%M:%S')

# 随机选择起降地和航司
def generate_flight_record():
    departure, arrival = random.choice(airport_pairs)
    airline = random.choice(airlines)
    departure_region = random.choice(regions)
    arrival_region = random.choice(regions)
    fleet_size = aircraft_fleet[airline]
    operation_area = f"{random.choice(regions)}-{random.choice(regions)}"
    
    return [
        generate_random_datetime(),
        departure,
        arrival,
        airline,
        departure_region,
        arrival_region,
        fleet_size,
        operation_area
    ]

# 生成数据
flight_records = [generate_flight_record() for _ in range(100)]

# 写入CSV文件
header = [
    '时间', '起飞地', '降落地', '航司', '起飞地区域', '降落地区域', '航司机队数目', '航司运营区域'
]

with open('flight_record.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(flight_records)

print("数据生成完成，已保存为 flight_record.csv")
