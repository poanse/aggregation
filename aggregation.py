import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from datetime import datetime

# To do:
# 1) generation
# 2) adaptive time step (dt)
# 3) differentiate between methods for user and auxiliary methods
# 4) add B calculation


class Buckets():
    def __init__(self, N, N0, dt, steps):
        self.N = N
        self.N0 = N0
        self.dt = dt
        self.steps = steps
        self.particle_number = np.zeros(N)
        self.particle_mass   = np.zeros(N)
        self.unit_mass       = np.zeros(N)
        self.number_grad     = np.zeros(N)
        self.mass_grad       = np.zeros(N)
        self.targets = np.zeros([N,N], dtype = np.int8)
        self.particle_number[0] = self.N0
        self.particle_mass[0]   = self.N0

    def set_b(self, B):
        self.B = B

    def update_unit_mass(self):
        mask = (self.particle_number != 0)
        self.unit_mass[mask] = self.particle_mass[mask] / self.particle_number[mask]

    def update(self):
        self.particle_number += self.number_grad * self.dt
        self.particle_mass += self.mass_grad * self.dt
        self.number_grad = np.zeros(self.N)
        self.mass_grad = np.zeros(self.N)

    def update_target(self,i,j):
        new_mass = self.unit_mass[i] + self.unit_mass[j]
        k = self.targets[i,j]
        while new_mass > self.max_mass[k]:
            k += 1
        while new_mass < self.min_mass[k]:
            k -= 1
        self.targets[i,j] = k

    def step(self):
        N = len(self.particle_number)
        self.update_unit_mass()
        for i in range(N):
            for j in range(i,N):
                if i == j:
                    alpha = 0.5
                else:
                    alpha = 1
                gain = alpha * self.particle_number[i] * self.particle_number[j] * self.B[i,j]
                if gain != 0:
                    self.update_target(i,j)
                    k = self.targets[i,j]
                
                    self.number_grad[k] += gain
                    self.number_grad[i] -= gain
                    self.number_grad[j] -= gain
                    
                    gain_i = gain * self.unit_mass[i]
                    gain_j = gain * self.unit_mass[j]
                
                    self.mass_grad[k] += gain_i + gain_j
                    self.mass_grad[i] -= gain_i
                    self.mass_grad[j] -= gain_j
        self.update()

    def run(self, blank=False):
        if blank == False:
            time = datetime.now().replace(microsecond=0)
            t = 0
            with open('./file_'+str(time)+'.txt','w+') as out:
                # Write header with parameters
                out.write('{};{};{};\n'.format(self.N,self.N0,self.dt))
                self.write_array(out, buckets.min_mass)
                self.write_array(out, buckets.max_mass)
                out.write('\n')
                # Write t0
                self.write_step(out, t)
                for i in tqdm(range(self.steps)):
                    self.step()
                    t += self.dt
                    # Write each step
                    self.write_step(out, t)
        elif blank == True:
            print('Blank run: results will not be saved.')
            for i in tqdm(range(self.steps)):
                self.step()
        else:
            raise ValueError()

    def write_array(self, out, array):
        for element in array:
            out.write('{};'.format(element))
        out.write('\n')
        
    def write_step(self, out, t):
        out.write('{};\n'.format(t))
        self.write_array(out, self.particle_number)
        self.write_array(out, self.particle_mass)

    def set_range(self, n_max, method='log'):
        if method == 'log':
            ### we will have linear logarithmic step
            # max_mass increases in geometric progression with coefficient alpha
            alpha = np.log10(n_max)
            alpha = alpha / (buckets.N - 1)
            alpha = 10**alpha

            array_max = []
            for i in range(self.N - 1):
                if i == 0:
                    array_max.append(1)
                else:
                    last_value = array_max[-1]
                    array_max.append(max(last_value + 1, ceil(last_value * alpha)))
            array_max.append(np.inf)
            array_max = np.array(array_max)
            array_min = np.append(1, array_max[:-1] + 1)
            array_min = np.array(array_min, dtype = np.int32)
            self.min_mass = array_min
            self.max_mass = array_max
        else:
            raise NameError("Not implemented yet. Default method is 'log'")


if __name__ == '__main__':
    #t_fin = 1
    #steps = int(round(t_fin / dt))
    buckets = Buckets(N=200,N0=10000,dt=0.005,steps=200)
    buckets.set_b(B = np.ones([buckets.N,buckets.N]) / buckets.N0 * 15)
    buckets.set_range(n_max=10**3)

    buckets.run(blank=True)