import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


N = 200
N0 = 10000
B = np.ones([N,N]) / N0 * 15
t_fin = 1
dt = 0.005
steps = 200
#steps = int(round(t_fin / dt))

class Buckets():
    def __init__(self, N):
        self.N = N
        self.particle_number = np.zeros(N)
        self.particle_mass = np.zeros(N)
        self.unit_mass = np.zeros(N)
        self.number_grad = np.zeros(N)
        self.mass_grad = np.zeros(N)
    def update_unit_mass(self):
        mask = (self.particle_number != 0)
        self.unit_mass[mask] = self.particle_mass[mask] / self.particle_number[mask]
    def update(self, dt):
        self.particle_number += self.number_grad * dt
        self.particle_mass += self.mass_grad * dt
        self.number_grad = np.zeros(self.N)
        self.mass_grad = np.zeros(self.N)

def update_target(buckets,targets,i,j):
    new_mass = buckets.unit_mass[i] + buckets.unit_mass[j]
    k = targets[i,j]
    while new_mass > buckets.max_mass[k]:
        k += 1
    while new_mass < buckets.min_mass[k]:
        k -= 1
    targets[i,j] = k

def step(buckets, targets, B, dt):
    N = len(buckets.particle_number)
    buckets.update_unit_mass()
    for i in range(N):
        for j in range(i,N):
            if i == j:
                alpha = 0.5
            else:
                alpha = 1
                
            gain = alpha * buckets.particle_number[i] * buckets.particle_number[j] * B[i,j]
            if gain != 0:
                update_target(buckets,targets,i,j)
                k = targets[i,j]
            
                buckets.number_grad[k] += gain
                buckets.number_grad[i] -= gain
                buckets.number_grad[j] -= gain
                
                gain_i = gain * buckets.unit_mass[i]
                gain_j = gain * buckets.unit_mass[j]
            
                buckets.mass_grad[k] += gain_i + gain_j
                buckets.mass_grad[i] -= gain_i
                buckets.mass_grad[j] -= gain_j
                
    buckets.update(dt)

def write_array(out, array):
    for element in array:
        out.write('{};'.format(element))
    out.write('\n')
    
def write_step(out, t, buckets):
    out.write('{};\n'.format(t))
    write_array(out, buckets.particle_number)
    write_array(out, buckets.particle_mass)

buckets = Buckets(N)

array_min = []
i = 0
for di in range(1, int(N/10) + 1):
    for j in range(10):
        i += di
        array_min.append(i)
        
array_min = np.array(array_min, dtype = np.int32)
array_max = np.append(array_min[1:] - 1, np.inf)
array_max = np.array(array_max)

buckets.min_mass = array_min
buckets.max_mass = array_max

buckets.particle_number[0] = N0
buckets.particle_mass[0] = N0

targets = np.zeros([N,N], dtype = np.int8)

time = datetime.now().replace(microsecond=0)
t = 0
with open('./file_'+str(time)+'.txt','w+') as out:
    # write header with parameters
    out.write('{};{};{};\n'.format(N,N0,dt))
    write_array(out, buckets.min_mass)
    write_array(out, buckets.max_mass)
    out.write('\n')
    # write t0
    write_step(out, t, buckets)
    for i in tqdm(range(steps)):
        step(buckets, targets, B, dt)
        t += dt
        # write each step
        write_step(out, t, buckets)
    
#for i in range(N):
    #print(buckets.min_mass[i], buckets.max_mass[i])
#print(buckets.particle_number.sum())

#plt.bar(buckets.unit_mass, np.append(buckets.particle_mass[:-1] / (buckets.max_mass[:-1] - buckets.min_mass[:-1] + 1), buckets.particle_mass[-1] ))
#plt.show()
