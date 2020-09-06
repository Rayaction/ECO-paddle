import os


# data_dir = 'data/hmdb_data_demo/'
data_dir = 'data/data48916/'

train_data = os.listdir(data_dir + 'train_split01')
train_data = [x for x in train_data if not x.startswith('.')]
print(len(train_data))

test_data = os.listdir(data_dir + 'test_split01')
test_data = [x for x in test_data if not x.startswith('.')]
print(len(test_data))

val_data = os.listdir(data_dir + 'val_split01')
val_data = [x for x in val_data if not x.startswith('.')]
print(len(val_data))

f = open('data/train_01.list', 'w')
for line in train_data:
    f.write(data_dir + 'train_split01/' + line + '\n')
f.close()
f = open('data/test_01.list', 'w')
for line in test_data:
    f.write(data_dir + 'test_split01/' + line + '\n')
f.close()
# f = open('data/val1.list', 'w')
# for line in val_data:
#     f.write(data_dir + 'val/' + line + '\n')
# f.close()

