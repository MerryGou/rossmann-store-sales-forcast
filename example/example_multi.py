import sys
sys.path.append('/home/goujiatong/FT_LM/business_model/auto_forecast-main')

# standard data manipulation imports
import pandas as pd
import tensorflow as tf
# import internal package functions
from src.plotting import *
from src.data_processing import *
from src.modeling import SalesForecasting

import matplotlib.pyplot as plt
import matplotlib as mpl

value_col = 'sales'
diffed_value_col = f"{value_col}_differenced"
date_col = 'Date'
mean_freq = 'Y'
forecast_horizon = 12
model_list = ['LinearRegression', 'RandomForest', 'XGBoost', 'LSTM', 'ARIMA'] 


train = pd.read_csv('/home/goujiatong/FT_LM/business_model/ARIMA/input/train.csv', parse_dates=True, index_col='Date')
store = pd.read_csv('/home/goujiatong/FT_LM/business_model/ARIMA/input/store.csv')
test = pd.read_csv('/home/goujiatong/FT_LM/business_model/ARIMA/input/test.csv')
#train = pd.read_csv("../input/train.csv", parse_dates = True, low_memory = False, index_col = 'Date')
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekofYear'] = train.index.isocalendar().week  # 替换原来的weekofyear

train['SalesPerCustomer'] = train['Sales']/train['Customers']

train.isnull().sum()
train.fillna(0, inplace = True)
train = train[(train["Open"] != 0) & (train['Sales'] != 0)]

test.isnull().sum()
test.fillna(0, inplace = True)
#test = test[(test["Open"] != 0) & (test['Sales'] != 0)]

print("In total: ", train.shape)

train=train.drop(columns=train[(train.Open == 1) & (train.Sales == 0)].index)
#test=test.drop(columns=test[(test.Open == 1) & (test.Sales == 0)].index)

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

train_store.loc[:, ['StoreType', 'Sales', 'Customers', 'PromoOpen', 'CompetitionOpen']].groupby('StoreType').mean()


print(f"train_store columns: {train_store.columns}")  # 验证列名

train_store['Sales'] = train_store['Sales'] * 1.0
#test['Sales'] = test['Sales'] * 1.0

#sales_a_test = test.loc[test.Store == 2, ['Sales']].copy()
#sales_a_test['Date'] = sales_a_test.index.to_series()  # 将索引转换为Date列

sales_a = train_store.loc[train_store.Store == 2].copy()
sales_a.to_csv(
    'sales_a.csv',
    index=False,
    encoding='utf-8-sig',  # 添加BOM标记，适合Excel打开中文文件
    chunksize=100000        # 分块写入，适合极大文件
)
'''
print("处理后的数据已保存为 'sales_a.csv'")
saved_data = pd.read_csv('sales_a.csv',encoding='utf-8-sig')
print("保存文件的基本信息:")
print(f"行数: {saved_data.shape[0]}, 列数: {saved_data.shape[1]}")
print(f"列名: {list(saved_data.columns)}")
print("\n前几行数据:")
print(saved_data.head())
'''
sales_a = train_store.loc[train_store.Store == 2, ['Date','Sales', 'Customers', 'Promo']].copy()
#sales_a['Date'] = sales_a.index.to_series()  # 将索引转换为Date列

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
'''
test_data = aggregate_by_time(
    data=sales_a_test,
    date_col='Date',  # 显式指定日期列
    resample_freq='D',
    aggregate='sum'
)
'''






import os
import json

# 转换日期格式
daily_data['Date'] = daily_data['Date'].astype(str)

# 保存JSON
os.makedirs(os.path.join(os.path.dirname(__file__), 'output'), exist_ok=True)
output_path = os.path.join(os.path.dirname(__file__), 'output/daily_sales_store_2.json')
daily_data[['Date', 'Sales']].to_json(
    output_path,
    orient='records',
    force_ascii=False
)

print(f"数据已保存至 {output_path}")


train_split = int(len(daily_data) * 0.8)
multi_data = daily_data[['Sales', 'Customers','Promo']].values
multi_data_mean = multi_data[:train_split].mean(axis =0)
multi_data_std = multi_data[:train_split].std(axis =0)
multi_data  = (multi_data - multi_data_mean)/ multi_data_std

#test_data = test_data['Sales'].values


#test_data_mean = test_data[:].mean()
#test_data_std = test_data[:].std()
#test_data  = (test_data - test_data_mean)/ test_data_std


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

history = 60
future_target = 6
STEP = 2

steps = 200

EPOCHS =20
batch_size = 32
buffer_size = 1000

x_train_ss , y_train_ss = mutlivariate_data(multi_data , multi_data[:, 0], 0, train_split, history,future_target, STEP , single_step = True)

x_val_ss , y_val_ss = mutlivariate_data(multi_data , multi_data[:,0] , train_split , None , history ,future_target, STEP, single_step = True)

print(x_train_ss.shape , y_train_ss.shape)


train_ss = tf.data.Dataset.from_tensor_slices((x_train_ss, y_train_ss))
train_ss = train_ss.cache().shuffle(buffer_size).batch(batch_size).repeat()

val_ss = tf.data.Dataset.from_tensor_slices((x_val_ss, y_val_ss))
val_ss = val_ss.cache().shuffle(buffer_size).batch(batch_size).repeat()

print(train_ss)
print(val_ss)

single_step_model = tf.keras.models.Sequential()

single_step_model.add(tf.keras.layers.LSTM(32, input_shape = x_train_ss.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))
single_step_model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mae')
single_step_model_history = single_step_model.fit(train_ss, epochs = EPOCHS , 
                                                  steps_per_epoch =steps, validation_data = val_ss,
                                                  validation_steps = 50)
                                      
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

plot_loss(single_step_model_history , 'Single Step Training and validation loss')
plt.savefig('output/plots/single_step_loss.png')

def plot_time_series(plot_data, delta , title):
  labels = ["History" , 'True Future' , 'Model Predcited']
  marker = ['.-' , 'rx' , 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])

  if delta:
    future = delta
  else:
    future = 0
  plt.title(title)
  for i , x in enumerate(plot_data):
    if i :
      plt.plot(future , plot_data[i] , marker[i], markersize = 10 , label = labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label = labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5) *2])

  plt.xlabel('Time_Step')
  return plt


def create_time_steps(length):
  return list(range(-length,0))

# plot time series and predicted values

for idx, (i, j) in enumerate(val_ss.take(5)):
    # 创建输出目录
    os.makedirs('output/plots', exist_ok=True)
    
    # 生成唯一文件名
    filename = f'lstm_mulvariate_sample_{idx+1}.png'
    
    # 绘制并保存
    plot = plot_time_series([i[0][:, 1].numpy(), j[0].numpy(),
                    single_step_model.predict(i)[0]], future_target,
                   f'LSTM Prediction_MS - Sample {idx+1}')
    plot.savefig(os.path.join('output/plots', filename))
    plot.close()  # 防止内存泄漏
    
print('5个样本图表已保存至output/plots目录')

### function to plot time series data
future_target = 72 # 72 future values
x_train_multi, y_train_multi = mutlivariate_data(multi_data, multi_data[:, 0], 0,
                                                 train_split, history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = mutlivariate_data(multi_data, multi_data[:, 0],
                                             train_split, None, history,
                                             future_target, STEP)

print(x_train_multi.shape)
print(y_train_multi.shape)

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(buffer_size).batch(batch_size).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(batch_size).repeat()

#plotting function
def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  plt.grid()
  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()
  


for x, y in train_data_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]))

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72)) # for 72 outputs

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=steps,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

plot_loss(multi_step_history, 'Multi-Step Training and validation loss')
plt.savefig('output/plots/multi_step_loss.png')

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

