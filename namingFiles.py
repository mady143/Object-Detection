import os
dir_name = 'Sudha'
path = './data_set/train/' + dir_name
files = os.listdir(path)
i=1
for file in files:    
    os.rename(os.path.join(path, file), os.path.join(path, dir_name + '.' + str(i) + '.jpg'))
    i = i+1
