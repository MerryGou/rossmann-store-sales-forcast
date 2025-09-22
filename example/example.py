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

train['Sales'] = train['Sales'] * 1.0
#test['Sales'] = test['Sales'] * 1.0

#sales_a_test = test.loc[test.Store == 2, ['Sales']].copy()
#sales_a_test['Date'] = sales_a_test.index.to_series()  # 将索引转换为Date列

sales_a = train.loc[train.Store == 2, ['Sales']].copy()
print(f"sales_a index: {sales_a.index}")
sales_a['Date'] = sales_a.index.to_series()  # 将索引转换为Date列

print(f"sales_a columns: {sales_a.columns}")  # 验证列名
print(f"sales_a head: {sales_a.head()}")  # 验证列名

daily_data = aggregate_by_time(
    data=sales_a,
    date_col='Date',  # 显式指定日期列
    resample_freq='D',
    aggregate='sum'
)
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
output_path = os.path.join(os.path.dirname(__file__), 'output/daily_sales.json')
daily_data[['Date', 'Sales']].to_json(
    output_path,
    orient='records',
    force_ascii=False
)

print(f"数据已保存至 {output_path}")


train_split = int(len(daily_data) * 0.8)
uni_data = daily_data['Sales'].values
uni_data_mean = uni_data[:train_split].mean()
uni_data_std = uni_data[:train_split].std()
uni_data  = (uni_data - uni_data_mean)/ uni_data_std

#test_data = test_data['Sales'].values


#test_data_mean = test_data[:].mean()
#test_data_std = test_data[:].std()
#test_data  = (test_data - test_data_mean)/ test_data_std


def univariate_data(dataset, start_idx , end_idx , history_size, target_size):
  data = []
  labels = []
  start_idx  = start_idx + history_size
  if end_idx is None:
    end_idx = len(dataset)- target_size
  for i in range(start_idx , end_idx):
    idxs = range(i-history_size , i)
    pre_idx = range(i,i+target_size)
    data.append(np.reshape(dataset[idxs] , (history_size, 1))) ### reshape data
    labels.append(np.reshape(dataset[pre_idx] , (target_size, 1)))
  return np.array(data), np.array(labels)

uni_data_history = 20   ## last 20 values
uni_data_future = 1     ## future data

x_train_uni , y_train_uni = univariate_data(uni_data , 0 , train_split , uni_data_history , uni_data_future)

x_val_uni , y_val_uni = univariate_data(uni_data , train_split , None ,uni_data_history , uni_data_future)
print(x_train_uni.shape , y_train_uni.shape)
print(x_val_uni.shape , y_val_uni.shape)

print('Single window of history data' , x_train_uni[0])

print('Target to predict ' , y_train_uni[0])

def create_time_steps(length):
  return list(range(-length,0))

### function to plot time series data

def plot_time_series(plot_data, delta, title):
    labels = ["History", 'True Future', 'Model Predicted']
    marker = ['.-', 'rx', 'go']
    
    # 历史时间步（从 -history_size 到 0）
    history_steps = list(range(-len(plot_data[0]), 0))
    
    # 未来时间步（从 1 到 delta）
    future_steps = list(range(1, delta + 1))
    
    plt.figure(figsize=(12, 6))
    plt.title(title)
    
    # 绘制历史数据
    plt.plot(history_steps, plot_data[0].flatten(), marker[0], label=labels[0])
    
    # 绘制真实未来值（四个点）
    plt.plot(future_steps, plot_data[1].flatten(), marker[1], markersize=8, label=labels[1])
    
    # 绘制模型预测值（四个点）
    if len(plot_data) > 2:
        plt.plot(future_steps, plot_data[2].flatten(), marker[2], markersize=8, label=labels[2])
    #plt.plot(future_steps, plot_data[2].flatten(), marker[2], markersize=8, label=labels[2])
    
    plt.legend()
    plt.xlim([history_steps[0], future_steps[-1] + 1])
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    
    # 添加垂直线标记当前时刻
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.text(0.1, plt.ylim()[1]*0.9, 'Current Time', rotation=90)
    
    return plt

plot_time_series([x_train_uni[0] , y_train_uni[0]] , uni_data_future , 'Sample Example')
plt.savefig('sample_example.png')

batch_size = 32
buffer_size = 1000

train_uni = tf.data.Dataset.from_tensor_slices((x_train_uni , y_train_uni))
train_uni = train_uni.cache().shuffle(buffer_size).batch(batch_size).repeat()

val_uni = tf.data.Dataset.from_tensor_slices((x_val_uni , y_val_uni))
val_uni = val_uni.cache().shuffle(buffer_size).batch(batch_size).repeat()

print(train_uni)
print(val_uni)

# 输入：时间步长*特征数
lstm_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(16 , input_shape = x_train_uni.shape[-2:]), 
                                         tf.keras.layers.Dense(uni_data_future)])

lstm_model.compile(optimizer = 'adam', loss = 'mae')

steps = 200

EPOCHS =50

lstm_model.fit(train_uni , epochs = EPOCHS, steps_per_epoch = steps ,
               validation_data = val_uni, validation_steps = 50)
# 在循环中添加索引
for idx, (i, j) in enumerate(val_uni.take(5)):
    # 创建输出目录
    os.makedirs('output/plots', exist_ok=True)
    
    # 生成唯一文件名
    filename = f'lstm_univariate_sample_{idx+1}.png'
    
    # 绘制并保存
    plot = plot_time_series(
        [i[0].numpy(), j[0].numpy(), lstm_model.predict(i)[0].squeeze()],
        uni_data_future,
        f'LSTM Prediction - Sample {idx+1}'
    )
    plot.savefig(os.path.join('output/plots', filename))
    plot.close()  # 防止内存泄漏
    
print('5个样本图表已保存至output/plots目录')

     
'''
print("数据已保存至output/daily_sales.xlsx")

daily_data.plot(x='Date', y='Sales', figsize=(12,6))
plt.title('daily sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('daily_sales.png')
plt.show()
'''



