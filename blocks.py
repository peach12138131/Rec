"""基于tensorflow搭建模型"""
import tensorflow as tf
from tensorflow.keras import layers, models

try:
    from tensorflow.keras.saving import register_keras_serializable
except:
    # 对于较旧版本的TensorFlow
    from tensorflow.keras.utils import register_keras_serializable

# 添加这个装饰器来注册模型
@register_keras_serializable()
class ResidualBlock(layers.Layer):
    def __init__(self, units, dropout_rate=0.5, activation='relu', use_batch_norm=True):
        """
        残差块的实现
        
        参数:
        units: 全连接层的输出维度
        dropout_rate: Dropout的比率，
        activation: 激活函数，默认'relu'
        use_batch_norm: 是否使用批归一化，默认True

        使用： # 自定义配置
            residual_block = ResidualBlock(
                units=64,
                dropout_rate=0.2,
                activation='selu',
                use_batch_norm=True
            )

        """
        super(ResidualBlock, self).__init__()
        
        # 定义两个全连接层，每层的输出维度都是units
        # dense1是第一个全连接层，用于对输入进行线性变换
        self.dense1 = layers.Dense(units)
        # dense2是第二个全连接层，用于对第一层的输出进行二次变换
        self.dense2 = layers.Dense(units)
        
        # 如果输入维度与输出维度不同，需要一个线性映射
        self.shortcut = None
        
        # 批归一化层
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = layers.BatchNormalization()
            self.bn2 = layers.BatchNormalization()
        
        # Dropout层
        self.dropout = layers.Dropout(dropout_rate)
        
        # 激活函数
        self.activation = layers.Activation(activation)
        
    def build(self, input_shape):
        # 如果输入维度与输出维度不同，创建shortcut映射
        if input_shape[-1] != self.dense1.units:
            self.shortcut = layers.Dense(self.dense1.units)
            
    def call(self, inputs, training=False):
        # 主路径
        x = self.dense1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        
        x = self.dense2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
            
        # shortcut路径
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
        else:
            shortcut = inputs
            
        # 残差连接
        x = layers.add([x, shortcut])
        
        # 最终激活
        output = self.activation(x)
        
        return output
    


# 添加这个装饰器来注册模型
@register_keras_serializable()
class CrossLayer(layers.Layer):
    def __init__(self, use_batch_norm=True, dropout_rate=0.1, activation='relu'):
        """
        参考残差网络设计的交叉层
        
        参数:
        use_batch_norm: 是否使用批归一化，默认True
        dropout_rate: Dropout比率，默认0.1
        activation: 激活函数，默认'relu'
        """
        super(CrossLayer, self).__init__()
        
        # 配置参数
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.activation_fn = activation
        
    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        
        # 主路径上的参数
        self.w = self.add_weight(
            name='w',
            shape=(feature_dim,),
            initializer='glorot_normal',
            trainable=True
        )
        
        self.b = self.add_weight(
            name='b',
            shape=(feature_dim,),
            initializer='zeros',
            trainable=True
        )
        
        # 批归一化层
        if self.use_batch_norm:
            self.batch_norm = layers.BatchNormalization()
            
        # Dropout层
        self.dropout = layers.Dropout(self.dropout_rate)
        
        # 激活函数
        self.activation = layers.Activation(self.activation_fn)
        
    def call(self, inputs, training=False):
        # 保存原始输入用于残差连接
        x_0 = inputs
        
        # 主路径: 特征交叉
        # 计算 x^T * w
        x_w = tf.reduce_sum(tf.multiply(inputs, self.w), axis=1, keepdims=True)
        # 计算交叉项 x_0 * (x^T * w)
        cross_term = tf.multiply(x_0, x_w)
        
        # 添加偏置
        x = cross_term + self.b
        
        # 批归一化
        if self.use_batch_norm:
            x = self.batch_norm(x, training=training)
            
        # 激活函数
        x = self.activation(x)
        
        # Dropout
        x = self.dropout(x, training=training)
        
        # 残差连接
        output = x + x_0
        
        return output
    
# 添加这个装饰器来注册模型
@register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embedding_dim=128, num_heads=8):
        """
        多头自注意力模块
        
        参数:
        embedding_dim: embedding向量的维度，默认128
        num_heads: 注意力头数，默认8
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # 确保embedding_dim能被num_heads整除
        assert embedding_dim % num_heads == 0
        
        # 每个头的维度
        self.depth = embedding_dim // num_heads
        
        # 定义线性变换层
        self.query_dense = layers.Dense(embedding_dim)
        self.key_dense = layers.Dense(embedding_dim)
        self.value_dense = layers.Dense(embedding_dim)
        
        self.output_dense = layers.Dense(embedding_dim)
        
    def split_heads(self, x, batch_size):
        """
        将输入分割成多个注意力头
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        
        # 线性变换
        query = self.query_dense(x)  # (batch_size, seq_len, embedding_dim)
        key = self.key_dense(x)      # (batch_size, seq_len, embedding_dim)
        value = self.value_dense(x)  # (batch_size, seq_len, embedding_dim)
        
        # 分割成多个头
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, depth)
        key = self.split_heads(key, batch_size)      # (batch_size, num_heads, seq_len, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len, depth)
        
        # 计算注意力得分
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        
        # 缩放注意力得分
        scale = tf.math.sqrt(tf.cast(self.depth, tf.float32))
        scaled_attention_logits = matmul_qk / scale
        
        # softmax获取注意力权重
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # 应用注意力权重
        scaled_attention = tf.matmul(attention_weights, value)
        
        # 重新整形
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
        
        # 最后的线性变换
        output = self.output_dense(concat_attention)
        
        return output, attention_weights

# 添加这个装饰器来注册模型
@register_keras_serializable()
class FeatureAttention(tf.keras.Model):
    def __init__(self, embedding_dim=128, num_heads=8, dropout_rate=0.1):
        """
        特征注意力模型
        
        参数:
        embedding_dim: embedding向量的维度，默认128
        num_heads: 注意力头数，默认8
        dropout_rate: dropout比率，默认0.1
        """
        super(FeatureAttention, self).__init__()
        
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # 前馈网络
        self.ffn = tf.keras.Sequential([
            layers.Dense(embedding_dim*4, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embedding_dim)
        ])
        
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, x, training=False):
        # 自注意力层
        attn_output, attention_weights = self.attention(x)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        
        return out2, attention_weights
    


