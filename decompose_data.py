import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    data_path = os.path.join(ROOT, "output", 'output.npy')
    out_dir = os.path.join(ROOT, "output", 'training_0000')
    data = np.load(data_path, allow_pickle=True)
    for i in range(len(data)):
        save_name = 'training_0000_' + str(i).zfill(4) + '.npz'
        np.savez(os.path.join(out_dir, save_name), **(data[i]))