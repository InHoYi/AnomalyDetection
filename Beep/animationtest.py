import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


data = pd.read_csv('')
data['measure_dtm'] = pd.to_datetime(data['measure_dtm'])
data.replace({"OFF": 0, "ON": 1}, inplace=True)


fig, ax = plt.subplots()
ln, = ax.plot([], [], 'b-')

def init():
    ax.set_xlim(0, 60)  
    ax.set_ylim(data['attribute_1_value'].min() - 5 , data['attribute_1_value'].max() + 5)
    ax.set_xlabel('Index of Time')  
    ax.set_ylabel('Attribute 1 Value') 
    return ln,

def update(frame_index):
    xdata.append(frame_index)
    ydata.append(data.loc[frame_index, 'attribute_1_value'])  # 인덱스로 직접 접근
    ln.set_data(xdata, ydata)
    # 동적으로 x축 범위 조정
    ax.set_xlim(max(0, frame_index - 30), frame_index + 30)
    return ln,

xdata, ydata = [], []
ani = FuncAnimation(fig, update, frames=range(len(data['measure_dtm'])), init_func=init, blit=True)

plt.show()
