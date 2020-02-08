import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

filename = 'file_2020-01-19 01:48:53.txt'

def Parse_agg_data(filename):
    
    with open('./'+filename, 'r') as file:
        data = file.read().split('\n')

    # split into header and data
    split  = data.index('') + 1
    header = data[:split]
    data   = data[split:-1]

    # read header
    N, N0, dt = [float(x) for x in header[0].split(';')[:-1]]
    mass_min  = [float(x) for x in header[1].split(';')[:-1]]
    mass_max  = [float(x) for x in header[2].split(';')[:-1]]

    # read data
    time   = [float(data[i][:-1]) for i in range(0,len(data),3)]
    time = np.round(time,10)
    number = [data[i].split(';')[:-1] for i in range(1,len(data),3)]
    mass   = [data[i].split(';')[:-1] for i in range(2,len(data),3)]
    for i, _ in enumerate(number):
            number[i] = [float(x) for x in number[i]]
            mass[i]   = [float(x) for x in mass[i]]

    mass_min = np.array(mass_min)
    mass_max = np.array(mass_max)
    number = np.array(number)
    mass   = np.array(mass)
    weight = np.zeros(number.shape)
    weight[number!=0] = mass[number!=0] / number[number!=0]
    width = mass_max - mass_min + 1

    # last bin is problematic
    #width[-1] = weight[-1,-1] - mass_min[-1]
    width[-1] = width[-2]
    return N, N0, dt, mass_min, mass_max, time, number, mass, weight, width

# average mass changes linearly with time
#print(mass.sum(axis=1)/number.sum(axis=1))
#plt.plot(time, mass.sum(axis=1)/number.sum(axis=1))
#plt.show()

N, N0, dt, mass_min, mass_max, time, number, mass, weight, width = Parse_agg_data(filename)

ax = plt.subplot(111)
plt.subplots_adjust(bottom=0.25)

# bins include append in order to include last bin
# last bin shoud be recalculated each time
plt.hist(weight[-1,:], bins=np.append(mass_min, weight[-1,-1]), weights = mass[-1,:]/width, label=str(time[-1]))

axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)
stime = Slider(axtime, 'Time', 0, len(time)-1, valinit = 0, valstep=1)

def update(val):
    index = int(stime.val)
    ax.cla()
    lastbin = max(weight[index,-1], mass_min[-1])
    ax.hist(weight[index,:], bins=np.append(mass_min, lastbin), weights = mass[index,:]/width, label=str(time[index]))
    ax.legend()
    plt.draw()

stime.on_changed(update)

ax.legend()
plt.show()
