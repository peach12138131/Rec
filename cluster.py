import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 新增：用于3D绘图
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import pandas as pd
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('./clusteroutputs', exist_ok=True)

# 加载航司向量
def load_vectors(file_path='./vectors/supplier_vectors.pkl'):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 寻找最佳聚类数K
def find_optimal_k(vectors, max_k=10):
    """
    使用轮廓系数(Silhouette Score)寻找最佳K值
    """
    sil_scores = []
    k_values = range(4, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)
        
        # 计算轮廓系数
        sil_score = silhouette_score(vectors, cluster_labels)
        sil_scores.append(sil_score)
        print(f"K = {k}, 轮廓系数 = {sil_score:.4f}")
    
    # 找出轮廓系数最高的K值
    best_k = k_values[np.argmax(sil_scores)]
    
    # 绘制轮廓系数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sil_scores, 'o-')
    plt.xlabel('聚类数 (K)')
    plt.ylabel('轮廓系数')
    plt.title('不同K值的轮廓系数')
    plt.grid(True)
    plt.savefig('./clusteroutputs/silhouette_scores.png')
    plt.close()
    
    return best_k

# 执行聚类分析
def cluster_analysis(data, k):
    """
    对供应商向量进行聚类分析
    """
    vectors = data['vectors']
    ids = data['ids']
    
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    
    # 创建结果DataFrame，确保supplier_id为字符串类型
    results = pd.DataFrame({
        'supplier_id': [str(id_) for id_ in ids],
        'cluster': cluster_labels
    })
    
    # 统计每个聚类的大小
    cluster_sizes = results['cluster'].value_counts().sort_index()
    print("\n各聚类大小:")
    for cluster_id, size in cluster_sizes.items():
        print(f"聚类 {cluster_id}: {size} 个航司")
    
    # 使用t-SNE进行2D降维可视化
    tsne_2d = TSNE(n_components=2, random_state=42)
    vectors_2d = tsne_2d.fit_transform(vectors)
    
    # 使用t-SNE进行3D降维可视化
    tsne_3d = TSNE(n_components=3, random_state=42)
    vectors_3d = tsne_3d.fit_transform(vectors)
    
    # 将降维结果添加到DataFrame
    results['x'] = vectors_2d[:, 0]
    results['y'] = vectors_2d[:, 1]
    results['x_3d'] = vectors_3d[:, 0]
    results['y_3d'] = vectors_3d[:, 1]
    results['z_3d'] = vectors_3d[:, 2]
    
    return results, kmeans.cluster_centers_

# 可视化聚类结果 (2D)
def visualize_clusters(results, k):
    """
    可视化聚类结果
    """
    plt.figure(figsize=(12, 10))
    
    # 为每个聚类设置不同的颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, k))
    
    for cluster_id in range(k):
        cluster_data = results[results['cluster'] == cluster_id]
        plt.scatter(cluster_data['x'], cluster_data['y'], s=50, c=[colors[cluster_id]], 
                   label=f'cluster {cluster_id} ({len(cluster_data)} suppliers)')
    
    plt.title('cluster_res')
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.legend()
    plt.grid(True)
    plt.savefig('./clusteroutputs/cluster_visualization.png')
    plt.close()

# 新增：3D可视化聚类结果
def visualize_clusters_3d(results, k):
    """
    3D可视化聚类结果
    """
    # 为每个聚类设置不同的颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, k))
    
    # 从不同角度生成3D图
    angles = [0, 45, 90, 135]
    for angle in angles:
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_id in range(k):
            cluster_data = results[results['cluster'] == cluster_id]
            ax.scatter(cluster_data['x_3d'], cluster_data['y_3d'], cluster_data['z_3d'],
                      s=50, c=[colors[cluster_id]], 
                      label=f'cluster {cluster_id} ({len(cluster_data)} suppliers)')
        
        ax.view_init(elev=30, azim=angle)
        ax.set_title(f'cluster_res_3d_{angle}')
        ax.set_xlabel('t-SNE1')
        ax.set_ylabel('t-SNE2')
        ax.set_zlabel('t-SNE3')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./clusteroutputs/cluster_visualization_3d_{angle}.png')
        plt.close()

# 分析每个聚类的代表性供应商
def analyze_clusters(results, data, top_n=5):
    """
    分析每个聚类的特点和代表性供应商
    """
    cluster_ids = results['cluster'].unique()
    vector_dict = data['vector_dict']
    
    # 加载特征数据，确保supplier_id为字符串类型
    supplier_features_path = './vectors/unique_supplier_features.pkl'
    if os.path.exists(supplier_features_path):
        try:
            supplier_features = pd.read_pickle(supplier_features_path)
            supplier_features['supplier_id'] = supplier_features['supplier_id'].astype(str)
            has_features = True
        except Exception as e:
            print(f"加载特征数据时出错: {e}")
            has_features = False
    else:
        has_features = False
    
    cluster_analysis = {}
    
    for cluster_id in sorted(cluster_ids):
        # 获取该聚类中的所有供应商
        cluster_suppliers = results[results['cluster'] == cluster_id]['supplier_id'].values
        
        if has_features:
            try:
                # 合并特征数据
                cluster_info = pd.DataFrame({'supplier_id': cluster_suppliers})
                cluster_info = cluster_info.merge(supplier_features, on='supplier_id', how='left')
                
                # 分析该聚类的特点
                country_distribution = cluster_info['orig_country_name'].value_counts().head(10)
                area1_distribution = cluster_info['orig_main_area_1'].value_counts().head(10)
                area2_distribution = cluster_info['orig_main_area_2'].value_counts().head(10)
                area3_distribution = cluster_info['orig_main_area_3'].value_counts().head(10)
                
                print(f"\n聚类 {cluster_id} ({len(cluster_suppliers)}个航司)分析:")
                print(f"主要国家分布: {country_distribution.to_dict()}")
                print(f"主要区域1分布: {area1_distribution.to_dict()}")
                print(f"主要区域2分布: {area2_distribution.to_dict()}")
                print(f"主要区域3分布: {area3_distribution.to_dict()}")
            except Exception as e:
                print(f"分析聚类 {cluster_id} 时出错: {e}")
        
        # 找出该聚类的代表性供应商
        representatives = []
        for supplier_id in cluster_suppliers:
            representatives.append(supplier_id)
            if len(representatives) >= top_n:
                break
        
        print(f"代表性航司: {representatives}")
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_suppliers),
            'representatives': representatives
        }
    
    return cluster_analysis

# 主函数
def main():
    try:
        # 加载航司向量
        print("加载航司向量数据...")
        data = load_vectors()
        print(f"成功加载 {len(data['vectors'])} 个航司向量")
        
        # 寻找最佳K值
        print("\n寻找最佳聚类数K...")
        best_k = find_optimal_k(data['vectors'])
        print(f"\n最佳聚类数K = {best_k}")
        
        # 使用最佳K值执行聚类
        print(f"\n使用K = {best_k}进行聚类分析...")
        results, cluster_centers = cluster_analysis(data, best_k)
        
        # 可视化聚类结果
        print("\n生成聚类可视化图...")
        visualize_clusters(results, best_k)  # 生成2D可视化
        visualize_clusters_3d(results, best_k)  # 新增：生成3D可视化
        
        # 分析每个聚类
        print("\n分析每个聚类的特点...")
        cluster_analysis_results = analyze_clusters(results, data)
        
        # 保存聚类结果
        results.to_csv('./clusteroutputs/supplier_clusters.csv', index=False)
        print("\n聚类结果已保存至 outputs/supplier_clusters.csv")
        
        # 计算聚类质量
        print("\n评估聚类质量...")
        silhouette_avg = silhouette_score(data['vectors'], results['cluster'])
        print(f"整体轮廓系数 = {silhouette_avg:.4f}")
        
        print("\n聚类分析完成!")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")

if __name__ == "__main__":
    main()