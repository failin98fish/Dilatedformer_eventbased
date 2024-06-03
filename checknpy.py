import numpy as np

def view_npy_file(file_path, num_rows):
    data = np.load(file_path)
    print(data[:num_rows])

# 使用示例
file_path = "/root/Deblurring-Low-Light-Images-with-Events/data/train/others/events/scene001_000.npy"
file_path1 = "/root/Deblurring-Low-Light-Images-with-Events/data/train/subnetwork1/events_voxel_grid/scene001_000.npy"
file_path2 = "/root/Deblurring-Low-Light-Images-with-Events/data/train/share/flow/scene001_000.npy"
num_rows = 100000000
print("++++++++++++++++++++++")

view_npy_file(file_path, num_rows)
# print("++++++++++++++++++++++")
# view_npy_file(file_path1, num_rows)
# print("++++++++++++++++++++++")
# view_npy_file(file_path2, num_rows)
# print("++++++++++++++++++++++")