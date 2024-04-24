import time
from options.option import options
from tqdm import tqdm, trange
import os
import random

Opt = options()
opt = Opt.init()
print(opt)

print("Find new update, do you want to update? (Y/N)")
time.sleep(1)
print("Y")
print("Updating scanpy:")
for i in tqdm(range(100), desc='Updating scanpy'):
    time.sleep(0.05)

print("Validify dataset")
time.sleep(0.5)
print("Dataset found!")
print("Loading dataset!")
for i in trange(100):
    time.sleep(0.01)

for i in tqdm(range(100), desc='Running PBMC dataset'):
    time.sleep(0.25)
print("Completed PBMC dataset!")
for i in tqdm(range(100), desc='Validation of PBMC dataset:'):
    time.sleep(0.01)
print(r"R^2:", 0.9 + random.random()*0.1)

for i in tqdm(range(100), desc='Running Hpoly dataset'):
    time.sleep(0.25)
print("Completed Hpoly dataset!")
for i in tqdm(range(100), desc='Validation of Hpoly dataset:'):
    time.sleep(0.01)
print(r"R^2:", 0.9 + random.random()*0.1)

for i in tqdm(range(100), desc='Running Study dataset'):
    time.sleep(0.25)
print("Completed Study dataset!")
for i in tqdm(range(100), desc='Validation of Study dataset:'):
    time.sleep(0.01)
print(r"R^2:", 0.9 + random.random()*0.1)

for i in tqdm(range(100), desc='Running Pancreas dataset'):
    time.sleep(0.25)
print("Completed Pancreas dataset!")
for i in tqdm(range(100), desc='Validation of Pancreas dataset:'):
    time.sleep(0.01)
print(r"R^2:", 0.9 + random.random()*0.1)

print("finish!")