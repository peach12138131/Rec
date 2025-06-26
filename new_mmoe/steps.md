



新流程

step1: 搞到数据，当前需要字段
``` csv
"dep_ts","arr_ts","supplier_name","supplier_id","supplier_country_name","pure_reg","aircraft_id","aircraft_model","aircraft_base","dep_icao","arr_icao","dep_latitude","dep_longitude","arr_latitude","arr_longitude","dep_city_id","arr_city_id","dep_city_name","arr_city_name","dep_country_id","arr_country_id","dep_country_name","arr_country_name","dep_area_id","arr_area_id","dep_area_name","arr_area_name","update_time","main_area_name","supplier_aircraft_scale_type","aircraft_models"
"1722802331401","1722809139000","Pacific Coast Jet","1815","United States","n862lg","11647","Phenom 300","KOAK","KASE","KOAK","39.22167","-106.86833","37.72167","-122.221664","1803690334658019330","1803691509839085570","Aspen","Oakland","165","165","United States","United States","1","1","North America","North America","20250226","North America,Central America,Oceania","13","Citation CJ2,Citation CJ3,Citation Excel,Citation M2,Citation Sovereign,Citation X,Citation XLS,Phenom 300,Pilatus PC-12"
"1702491243331","1702497149631","Jet Linx Aviation, LLC","1609","United States","n88af","15664","Citation XLS+","KSTL","KTUS","KSAT","32.116665","-110.941666","29.533333","-98.46833","1803691913377267714","1803690816793264129","Tucson","San Antonio","165","165","United States","United States","1","1","North America","North America","20250226","North America,Central America,South America","38","Challenger 300,Challenger 604,Challenger 605,Citation CJ3,Citation Latitude,Citation Sovereign,Citation X,Citation XLS,Citation XLS+,Falcon 2000EX,Falcon 7X,Falcon 900B,Falcon 900EX,Gulfstream G150,Gulfstream G200,Gulfstream G280,Hawker 800XP,Hawker 850XP,Learjet 45,Learjet 60,Nextant 400XT"
```

step2: 直接encoding.py,跳过负样本构造,进行特征处理:数字化，分桶，按区域分割数据等等
step3: 阅读model.py，知悉模型结构，掌握基础训练方式，能跑通无错代表模型正确可执行
step4. MMOE_train.py,正式训练脚本
        b.通过分区随机负采样生成负样本
        c.通过强化学习指明什么是明显不能学

step4.等待写：运行prepare_deploy.py,生成各个pkl,json特征文件,和supplier_vectors.pkl

step5.等待写：offline_deploy.py,进行离线推荐

