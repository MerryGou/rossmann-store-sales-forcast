import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from example_multistep import multi_step_plot  # 替换为你的模块名

# 设置 matplotlib 后端（对于无头环境很重要）
plt.switch_backend('Agg')

# 创建输出目录
output_dir = 'validation_predictions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
future_target = 20
# 加载模型和数据
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(future_target)) 
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# 3. 加载保存的权重
multi_step_model.load_weights('multi_step_model.weights.h5')

print("模型权重已成功加载")
# multi_step_model = tf.keras.models.load_model('your_model.h5')
# val_data_multi = ... # 你的验证数据

def main():
    # 确保模型和数据已加载
    
    
    for i, (x, y) in enumerate(val_data_multi.take(3)):
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 调用绘图函数
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
        
        # 添加标题
        plt.title(f'Validation Sample {i+1}')
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, f'validation_prediction_{i+1}.png'))
        
        # 关闭图形以释放内存
        plt.close()
    
    print(f"已保存3张验证集预测图片到'{output_dir}'目录")

if __name__ == "__main__":
    main()