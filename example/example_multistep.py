import sys
sys.path.append('/home/goujiatong/FT_LM/business_model/auto_forecast-main')
import os
# standard data manipulation imports
import pandas as pd
import tensorflow as tf
# import internal package functions
from src.plotting import *
from src.data_processing import *
from src.modeling import SalesForecasting

import matplotlib.pyplot as plt
import matplotlib as mpl
import random
def mutlivariate_data(dataset , target , start_idx , end_idx , history_size , target_size,
                      step ,  single_step = False):
  data = []
  labels = []
  start_idx = start_idx + history_size
  if end_idx is None:
    end_idx = len(dataset)- target_size
  for i in range(start_idx , end_idx ):
    idxs = range(i-history_size, i, step) ### using step
    data.append(dataset[idxs])
    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data) , np.array(labels)

def plot_loss(history , title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss , 'b' , label = 'Train Loss')
  plt.plot(epochs, val_loss , 'r' , label = 'Validation Loss')
  plt.title(title)
  plt.legend()
  plt.grid()
  plt.show()



def create_time_steps(length):
  return list(range(-length,0))

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  plt.grid()
  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b-o',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'r-o',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

if __name__ == '__main__':

  

  train = pd.read_csv('/home/goujiatong/FT_LM/business_model/ARIMA/input/train.csv', parse_dates=True, index_col='Date')
  store = pd.read_csv('/home/goujiatong/FT_LM/business_model/ARIMA/input/store.csv')
  #train = pd.read_csv("../input/train.csv", parse_dates = True, low_memory = False, index_col = 'Date')
  train['Year'] = train.index.year
  train['Month'] = train.index.month
  train['Day'] = train.index.day
  train['WeekofYear'] = train.index.isocalendar().week  # 替换原来的weekofyear
  train['SalesPerCustomer'] = train['Sales']/train['Customers']

  train.isnull().sum()
  train.fillna(0, inplace = True)
  train = train[(train["Open"] != 0) & (train['Sales'] != 0)]

  print("In total: ", train.shape)

  train=train.drop(columns=train[(train.Open == 1) & (train.Sales == 0)].index)

  store.isnull().sum()
  store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)
  store.fillna(0, inplace = True)
  train['Date'] = train.index
  train_store = pd.merge(train, store, how = 'inner', on = 'Store')

  print("train_store columns: ", train_store.columns)
  print("train_store In total: ", train_store.shape)


  train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + \
          (train_store.Month - train_store.CompetitionOpenSinceMonth)
      
  # Promo open time
  train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) + \
          (train_store.WeekofYear - train_store.Promo2SinceWeek) / 4.0  # 保持列名全小写

  # replace NA's by 0
  train_store.fillna(0, inplace = True)
  print(f"train_store columns: {train_store.columns}")  # 验证列名

  train_store['Sales'] = train_store['Sales'] * 1.0

  sales_a = train_store.loc[train_store.Store == 2].copy()
  '''
  sales_a.to_csv(
      'sales_a.csv',
      index=False,
      encoding='utf-8-sig',  # 添加BOM标记，
      chunksize=100000        # 分块写入，适合极大文件
  )

  print("处理后的数据已保存为 'sales_a.csv'")
  saved_data = pd.read_csv('sales_a.csv',encoding='utf-8-sig')
  print("保存文件的基本信息:")
  print(f"行数: {saved_data.shape[0]}, 列数: {saved_data.shape[1]}")
  print(f"列名: {list(saved_data.columns)}")
  print("\n前几行数据:")
  print(saved_data.head())
  '''
  sales_a = train_store.loc[train_store.Store == 2, ['Date','Sales', 'Customers', 'Promo']].copy()

  print(f"sales_a columns: {sales_a.columns}")  # 验证列名
  print(f"sales_a head: {sales_a.head()}")  # 验证列名

  daily_data = aggregate_by_time(
      data=sales_a,
      date_col='Date',  # 显式指定日期列
      resample_freq='D',
      aggregate='sum'
  )
  print(f"daily_data head: {daily_data.head()}")  # 
  print(f"daily_data tail: {daily_data.tail()}")  # 



  train_split = int(len(daily_data) * 0.8)
  multi_data = daily_data[['Sales', 'Customers','Promo']].values
  multi_data_mean = multi_data[:train_split].mean(axis =0)
  multi_data_std = multi_data[:train_split].std(axis =0)
  multi_data  = (multi_data - multi_data_mean)/ multi_data_std



  history = 80
  future_target = 20 
  STEP = 1

  steps = 31
  EPOCHS =100
  batch_size = 32
  buffer_size = 1000
  is_train = False

  x_train_multi, y_train_multi = mutlivariate_data(multi_data, multi_data[:, 0], 0,
                                                  train_split, history,
                                                  future_target, STEP)
  x_val_multi, y_val_multi = mutlivariate_data(multi_data, multi_data[:, 0],
                                              train_split, None, history,
                                              future_target, STEP)

  print(x_train_multi.shape)
  print(y_train_multi.shape)

  print(x_val_multi.shape)
  print(y_val_multi.shape)

  train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
  train_data_multi = train_data_multi.cache().shuffle(buffer_size).batch(batch_size).repeat()

  val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
  val_data_multi = val_data_multi.batch(batch_size).repeat()

  multi_step_model = tf.keras.models.Sequential()
  multi_step_model.add(tf.keras.layers.LSTM(32,
                                            return_sequences=True,
                                            input_shape=x_train_multi.shape[-2:]))
  multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
  multi_step_model.add(tf.keras.layers.Dense(future_target)) 

  multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
  if is_train:
      multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                                steps_per_epoch=steps,#单个epoch包含多少训练批次
                                                validation_data=val_data_multi,#每次验证使用的批次数量
                                                validation_steps=2)
      multi_step_model.save_weights('multi_step_model.weights.h5')
      print("模型权重已保存")
      plot_loss(multi_step_history, 'Multi-Step Training and validation loss')
      plt.savefig('output/plots/multi_step_loss.png')
  else:
      multi_step_model.load_weights('multi_step_model.weights.h5')
      print("模型权重已成功加载")
    


 
  output_dir = 'validation_predictions'
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  # 循环处理前3个验证样本
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



  train_sales = daily_data['Sales'].values[:train_split]
  naive_forecast_errors = np.abs(train_sales[1:] - train_sales[:-1])
  mae_naive = np.mean(naive_forecast_errors)

  full_mae = []
  full_mse = []
  full_mape = []
  full_mase = []

  for x, y in val_data_multi.take(3):
      pred = multi_step_model.predict(x)
      y_np = y.numpy()
      
      # 提取 Sales 对应的均值和标准差
      sales_mean = multi_data_mean[0]
      sales_std = multi_data_std[0]
      
      # 逆变换
      y_sales_original = y_np * sales_std + sales_mean
      pred_sales_original = pred * sales_std + sales_mean
      
      # 计算指标
      batch_mae = np.mean(np.abs(y_sales_original - pred_sales_original), axis=1)
      batch_mse = np.mean((y_sales_original - pred_sales_original)**2, axis=1)
      
      # 计算MAPE（但可能不可靠）
      batch_mape = 100 * np.mean(
          np.abs((y_sales_original - pred_sales_original) / (y_sales_original + 1e-8)), 
          axis=1
      )
      
      # 计算MASE（更可靠的指标）
      batch_mase = batch_mae / mae_naive
      
      full_mae.extend(batch_mae)
      full_mse.extend(batch_mse)
      full_mape.extend(batch_mape)
      full_mase.extend(batch_mase)

  # 输出指标
  print("\n全体验证集指标:")
  print(f"样本数: {len(full_mae)}")
  print(f"MAE 均值: {np.mean(full_mae):.4f} ± {np.std(full_mae):.4f}")
  print(f"MSE 均值: {np.mean(full_mse):.4f} ± {np.std(full_mse):.4f}")
  print(f"MAPE均值: {np.mean(full_mape):.2f}% ± {np.std(full_mape):.2f}%")
  print(f"MASE均值: {np.mean(full_mase):.4f} ± {np.std(full_mase):.4f}")

  # 额外：检查MAPE异常高的样本
  if np.max(full_mape) > 1000:  # 如果MAPE异常高
      high_mape_indices = np.argsort(full_mape)[-5:]
      print("\nMAPE最高的5个样本:")
      for i, idx in enumerate(high_mape_indices):
          print(f"样本 {idx}: 真实值={y_sales_original[idx]}, 预测值={pred_sales_original[idx]}, MAPE={full_mape[idx]:.2f}%")