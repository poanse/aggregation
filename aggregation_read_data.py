import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class Experiment(): # container for results of the experiment
    def __init__(self, N, N0, dt, mass_min, mass_max, time, number, mass, weight, width):
        self.N        = N
        self.N0       = N0
        self.dt       = dt
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.time     = time
        self.number   = number
        self.mass     = mass
        self.weight   = weight
        self.width    = width

def Parse_agg_data(filename): # experiment data parser
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
    time   = np.round(time,10)
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

    experiment = Experiment(N, N0, dt, mass_min, mass_max, time, number, mass, weight, width)
    return experiment

# average mass changes linearly with time
#print(mass.sum(axis=1)/number.sum(axis=1))
#plt.plot(time, mass.sum(axis=1)/number.sum(axis=1))
#plt.show()

def draw_plots(ax, experiment, step, graph='plot', normalize=False):
    ### Only nonzero bins should be visualized and their number should be the same at every step to make graph readable
    max_mass = max(experiment.mass[-1,:])
    for i, mass in enumerate(experiment.mass[-1,:]):
        if mass >= 0.01*max_mass:
            last_nonzero_bin = i

    bars_position = (experiment.mass_max[:-1] + experiment.mass_min[:-1]) / 2 # mass_max for last bar is np.inf
    last_bar_position = (experiment.weight[step,-1] + experiment.mass_min[-1]) / 2 
    bars_position = np.append(bars_position, last_bar_position)

    bars_height = experiment.mass[step,:] / experiment.width # total mass in a bin normalized to its width

    alpha = 1
    if normalize == True:
        alpha = 1 / max(bars_height[:])

    label = 'time = '+str(experiment.time[step])

    if graph == 'bar':
        # bars width is needed only for barplot 
        bars_width = experiment.mass_max[:-1] - experiment.mass_min[:-1] + 1
        last_bar_width = experiment.weight[step,-1]
        bars_width = np.append(bars_width, last_bar_width)

        ax.bar(x      = bars_position[:last_nonzero_bin], 
               height = bars_height[:last_nonzero_bin] * alpha, 
               width  = bars_width[:last_nonzero_bin],
               label  = label)
    else:
        ax.plot(bars_position[:last_nonzero_bin],
                bars_height[:last_nonzero_bin] * alpha,
                label = 'time = ' + str(experiment.time[step]))


def draw_aggregation_data(filename,graph,interactive):
    experiment = Parse_agg_data(filename)

    ax = plt.subplot(111)
    plt.subplots_adjust(bottom=0.25)

    if interactive == True:
        draw_plots(ax, experiment, step=-1, graph=graph, normalize=False)
        axtime = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgoldenrodyellow')
        stime = Slider(axtime, 'Step', 0, len(experiment.time)-1, valinit = 0, valstep=1)
        def update(val):
            ax.cla()
            draw_plots(ax, experiment, step=int(stime.val), graph=graph, normalize=False)
            ax.legend()
            plt.draw()

        stime.on_changed(update)
        ax.legend()
        plt.show()
    elif interactive == False:
        for i, time in enumerate(experiment.time):
            if i % int(len(experiment.time) / 10) == 0:
                draw_plots(ax, experiment, i, graph, normalize=True)
        ax.legend()
        plt.show()
    else:
        raise ValueError()

if __name__ == '__main__':
    filename = 'file_2020-02-08 18:45:18.txt'
    draw_aggregation_data(filename, graph='plot', interactive=True)