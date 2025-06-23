



新流程

step1: 特征构造 (在正样本上完成)
step2: 直接encoding,跳过负样本构造
step3: 训练,运行contrastive_train.py,进行对比训练
b.通过分区随机负采样生成负样本
c.通过强化学习指明什么是明显不能学

step4.运行prepare_deploy.py,生成各个pkl,json特征文件,和supplier_vectors.pkl

step5.运行offline_deploy.py,进行离线推荐

