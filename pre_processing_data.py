

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import string,cv2
import pathlib,json,os


current_directory_path = pathlib.Path(".").absolute()
current_directory_path

# Import json file 1
with open('C:/Users/nguye/OneDrive/Code/handwirtting_text2/datasets/raw/labels_0916.json',encoding='utf8') as json_file:
    label_data = json.load(json_file)

label_data['0000_tests.png']


char_list= set()
for label in label_data.values():
    char_list.update(set(label))
char_list=sorted(char_list)
print(char_list)

"".join(char_list)


"""
Mã hóa chuỗi lấy index từ char_list
"""
def encode_to_labels(txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print("No found in char_list :", char)
        
    return dig_lst


raw_folder = 'C:/Users/nguye/OneDrive/Code/handwirtting_text2/datasets/raw/'

train_image_path = []

for item in pathlib.Path(raw_folder).glob('**/*'):
    if item.is_file() and item.suffix not in [".json"]:
        train_image_path.append(str(item))



dict_filepath_label={}
raw_data_path = pathlib.Path(os.path.join(raw_folder))
for item in raw_data_path.glob('**/*.*'):
    file_name=str(os.path.basename(item))
    if file_name != "labels_0916.json":
        label = label_data[file_name]
        dict_filepath_label[str(item)]=label



label_lens= []
for label in dict_filepath_label.values():
    label_lens.append(len(label))
max_label_len = max(label_lens)


all_image_paths = list(dict_filepath_label.keys())



import cv2
widths = []
heights = []
for image_path in all_image_paths:
    img = cv2.imread(image_path)
    (height, width, _) = img.shape
    heights.append(height)
    widths.append(width)


# In[17]:


min_height = min(heights)
max_height = max(heights)
min_width = min(widths)
max_width = max(widths)


# In[18]:


print(min_height, max_height, min_width, max_width)


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test = train_test_split(all_image_paths,test_size=0.2,random_state=1)


# In[20]:


training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []
resize_max_width=0
TIME_STEPS = 240


# Chúng ta sẽ resize lại bức ảnh, tuy nhiên ở bài trước đầu vào 1 từ vì vậy input khác bây giờ đầu vào của bài toán chúng ta câu vì chúng cần rezise lại (118,2167)



"""
Tạo tập dữ liệu training, mã hóa, độ dài đầu vào 
"""
i = 0
for train_img_path in X_train:
    img = cv2.cvtColor(cv2.imread(train_img_path), cv2.COLOR_BGR2GRAY)
    w, h = img.shape
    new_w = 118
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    img = img.astype('float32')
    
    
    if w < 118:
        add_zeros = np.full((118-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 2167:
        add_zeros = np.full((w, 2167-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
    if h > 2167 or w > 118:
        dim = (2167,118)
        img = cv2.resize(img, dim)
    
    if img.shape[1] > resize_max_width:
        resize_max_width = img.shape[1]
        
    img = cv2.subtract(255, img)
    img = np.expand_dims(img , axis = 2)
    img = img/255.0
    label = dict_filepath_label[train_img_path]
    orig_txt.append(label)   
    train_label_length.append(len(label))
    train_input_length.append(TIME_STEPS)
    training_img.append(img)
    training_txt.append(encode_to_labels(label)) 
    i+=1
    if (i%500 == 0):
        print ("has processed trained {} files".format(i))




import matplotlib.pyplot as plt
for i in range(5):
    plt.figure(figsize=(15,2))
    plt.imshow(training_img[i][:,:,0], cmap="gray")
    plt.show()


"""
Tạo tập dữ liệu validation, mã hóa, độ dài đầu vào
"""
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

i = 0
for train_img_path in X_test:
    img = cv2.cvtColor(cv2.imread(train_img_path), cv2.COLOR_BGR2GRAY)
    w, h = img.shape
    new_w = 118
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    img = img.astype('float32')
    
    

    if w < 118:
        add_zeros = np.full((118-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 2167:
        add_zeros = np.full((w, 2167-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
    if h > 2167 or w > 118:
        dim = (2167,118)
        img = cv2.resize(img, dim)
    
    if img.shape[1] > resize_max_width:
        resize_max_width = img.shape[1]
    
    img = cv2.subtract(255, img)
    img = np.expand_dims(img , axis = 2)
    img = img/255.0
    label = dict_filepath_label[train_img_path]
    valid_orig_txt.append(label)   
    valid_label_length.append(len(label))
    valid_input_length.append(TIME_STEPS)
    valid_img.append(img)
    valid_txt.append(encode_to_labels(label)) 
    i+=1
    if (i%500 == 0):
        print ("has processed trained {} files".format(i))


# In[24]:


for i in range(5):
    plt.figure(figsize=(15,2))
    plt.imshow(valid_img[i][:,:,0], cmap="gray")
    plt.show()




max_label_len = TIME_STEPS 



from tensorflow.keras.preprocessing.sequence import pad_sequences

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value = 0)
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = 0)



train_images = np.array(training_img)
valid_images = np.array(valid_img)

train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)


with open('train_images.npy', 'wb') as f:
    np.save(f, train_images)
with open('train_input_length.npy', 'wb') as f:
    np.save(f, train_input_length)
with open('train_label_length.npy', 'wb') as f:
    np.save(f, train_label_length)

with open('train_padded_label.npy', 'wb') as f:
    np.save(f, train_padded_txt)
with open('valid_padded_label.npy', 'wb') as f:
    np.save(f, valid_padded_txt)

    
with open('valid_images.npy', 'wb') as f:
    np.save(f, valid_images)
with open('valid_input_length.npy', 'wb') as f:
    np.save(f, valid_input_length)
with open('valid_label_length.npy', 'wb') as f:
    np.save(f, valid_label_length)