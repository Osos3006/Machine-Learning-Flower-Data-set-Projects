# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 23:37:25 2020

@author: Mohammed Abuelwafa 
"""
import numpy as np
#import os
import platform
#import PIL
import matplotlib as plt
import NearestNeighbor as NN
from PIL import Image
from numpy import asarray
from os import listdir
import cv2


'''
rmage = PIL.Image.open("./flower_photos/daisy/5547758_eea9edfd54_n.jpg")
new_image = rmage.resize((32,32))
print (rmage.size)
#plt.pyplot.imshow(rmage)
print (new_image.size)
plt.pyplot.imshow(new_image)
'''

'''
rmage = PIL.Image.open("./flower_photos/daisy/5547758_eea9edfd54_n.jpg")
data = plt.image.imread("./flower_photos/daisy/5547758_eea9edfd54_n.jpg")
print (data.dtype)
print (data.shape)
plt.pyplot.imshow(data)
plt.pyplot.show()
''' 

'''
# load image and convert to and from NumPy array
from PIL import Image
from numpy import asarray
# load the image
image = Image.open('opera_house.jpg')
# convert image to numpy array
data = asarray(image)
# summarize shape
print(data.shape)
# create Pillow image
image2 = Image.fromarray(data)
# summarize image details
print(image2.format)
print(image2.mode)
print(image2.size)

'''

'''
# load all images in a directory
from os import listdir
from matplotlib import image
# load all images in a directory
loaded_images = list()
for filename in listdir('images'):
	# load image
	img_data = image.imread('images/' + filename)
	# store loaded image
	loaded_images.append(img_data)
	print('> loaded %s %s' % (filename, img_data.shape))
'''
# load all images in a directory
def load_directory (f_name):
    #print ("in file " + f_name)
    loaded_images = {}
    labels_list = []
    loaded_images_list = []
    #sizes = []
    for filename in listdir('flower_photos/'+ f_name):
    	# load image
        img_data = plt.image.imread('flower_photos/'+ f_name +'/' + filename)
        new_img = cv2.resize(img_data, dsize=(64,64),interpolation=cv2.INTER_CUBIC )
        #print(type(img_data))
        	# store loaded image
        #loaded_images.append(img_data) 
        #new_img = img_data.resize((64,64))
        loaded_images[filename]= new_img
        #sizes.append(img_data.shape)
        #print('> loaded %s %s' % (filename, img_data.shape))
        sorted_loaded_images = dict( sorted(loaded_images.items(), key=lambda x: x[0].lower()) )     
    '''    
    sortedkeys=sorted(loaded_images.keys(), key=lambda x:x.lower())  
    print (sortedkeys)
    for i in sortedkeys:
        values = loaded_images[i]
        loaded_images_list.append(values)
    '''    
    
    for key in (sorted_loaded_images):
        #print(key)
        labels_list.append(f_name)
        loaded_images_list.append(loaded_images[key])
        
    ret = np.array(loaded_images_list) 
    labels = np.array(labels_list)
    #min_sz = min(sizes) #to get the minimum image size to size it accordingly..
    testing_batch = ret[-100:]
    testing_batch_labels = labels[-100:]
  #  print(testing_batch_labels)
  #  print(testing_batch_labels.shape)
    #plt.pyplot.figure()
    #plt.pyplot.imshow(testing_batch[0])
   # print (testing_batch.shape)
    training_batch = ret[:-100].copy()
    training_batch_labels = labels[:-100].copy()
    #print(training_batch.shape)
    
    return testing_batch,training_batch,testing_batch_labels,training_batch_labels


training_data = []
testing_data = []
training_labels = []
testing_labels=[]

#minis = np.empty(5, dtype=tuple)
#index = 0
#testing_batch, training_batch = load_directory('daisy')

for directoryname in listdir('flower_photos'):
    print ('in directory:' + directoryname)
    testing_batch, training_batch,testing_batch_labels,training_batch_labels = load_directory(directoryname)
    testing_data.extend(testing_batch) 
    testing_labels.extend(testing_batch_labels)
  #  plt.pyplot.imshow(testing_batch[0])
    training_data.extend(training_batch)
    training_labels.extend(training_batch_labels)
    #index = index + 1
    #a, mini = load_directory('daisy')
    
    
'''
    print (a.shape)
    print (minis[0])
    #plt.pyplot.imshow(img)
    b, minis[1] = load_directory('roses')
    print (minis[1])
    print (b.shape)
    a = np.append(a, b)
    print (a.shape) 
    '''

testing_data = np.array(testing_data)
training_data = np.array(training_data)
training_labels= np.array(training_labels)
testing_labels = np.array (testing_labels)
#print (testing_data.shape)
#print (testing_data)
#print (training_data.shape)
#print (training_data[0].shape)
#print (training_data[1].shape)

#print (testing_labels.shape)
#print (training_labels.shape)
#training_labels = np.concatenate(training_labels).ravel()
#print (training_labels.shape)
#print (testing_labels.shape)
#testing_labels = np.concatenate(testing_labels).ravel()
#print (training_data.shape)

#testing_data = np.concatenate(testing_data).ravel()


#m,n,r = testing_data.shape[:::3]
#testing_data.reshape()
'''
testing_data_new = testing_data[0, :, :, :,:]
for i in range(1, testing_data.shape[0]):
    testing_data_new = np.concatenate((testing_data_new, testing_data[i, :, :, :,:]), axis=1)
print (testing_data_new.shape)
'''
#print (testing_labels.shape)
#print (testing_labels)
np.random.seed(10)
#np.random.shuffle(testing_labels)
#np.random.shuffle(testing_data)
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a , b
    
testing_data,testing_labels=shuffle_in_unison(testing_data,testing_labels)
daisy,dandelion,roses,sunflowers,tulips = np.split(testing_data,5)
daisy_lbls,dandelion_lbls,roses_lbls,sunflowers_lbls,tulips_lbls=np.split(testing_labels,5)
training_data,training_labels=shuffle_in_unison(training_data,training_labels)

fold_1,fold_2,fold_3,fold_4,fold_5 = np.split(training_data,5)
fold_1_lbl,fold_2_lbl,fold_3_lbl,fold_4_lbl,fold_5_lbl = np.split(training_labels,5)

def generate_validation_fold (fold_1,fold_2,fold_3,fold_4,fold_5,fold_1_lbl,fold_2_lbl,fold_3_lbl,fold_4_lbl,fold_5_lbl, n):
    training_folds = []
    training_folds_labels=[]
    if (n==1):
        training_folds.extend(fold_2)
        training_folds.extend(fold_3)
        training_folds.extend(fold_4)
        training_folds.extend(fold_5)
        training_folds_labels.extend(fold_2_lbl)
        training_folds_labels.extend(fold_3_lbl)
        training_folds_labels.extend(fold_4_lbl)
        training_folds_labels.extend(fold_5_lbl)
        validation_fold = fold_1
        validation_fold_lbls= fold_1_lbl
    if (n==2):
        training_folds.extend(fold_1)
        training_folds.extend(fold_3)
        training_folds.extend(fold_4)
        training_folds.extend(fold_5)
        training_folds_labels.extend(fold_1_lbl)
        training_folds_labels.extend(fold_3_lbl)
        training_folds_labels.extend(fold_4_lbl)
        training_folds_labels.extend(fold_5_lbl)
        validation_fold = fold_2
        validation_fold_lbls= fold_2_lbl
    if (n==3):
        training_folds.extend(fold_1)
        training_folds.extend(fold_2)
        training_folds.extend(fold_4)
        training_folds.extend(fold_5)
        training_folds_labels.extend(fold_1_lbl)
        training_folds_labels.extend(fold_2_lbl)
        training_folds_labels.extend(fold_4_lbl)
        training_folds_labels.extend(fold_5_lbl)
        validation_fold = fold_3
        validation_fold_lbls= fold_3_lbl
    if (n==4):
        training_folds.extend(fold_1)
        training_folds.extend(fold_2)
        training_folds.extend(fold_3)
        training_folds.extend(fold_5)
        training_folds_labels.extend(fold_1_lbl)
        training_folds_labels.extend(fold_2_lbl)
        training_folds_labels.extend(fold_3_lbl)
        training_folds_labels.extend(fold_5_lbl)
        validation_fold = fold_4
        validation_fold_lbls= fold_4_lbl
    if (n==5):
        training_folds.extend(fold_1)
        training_folds.extend(fold_2)
        training_folds.extend(fold_3)
        training_folds.extend(fold_4)
        training_folds_labels.extend(fold_1_lbl)
        training_folds_labels.extend(fold_2_lbl)
        training_folds_labels.extend(fold_3_lbl)
        training_folds_labels.extend(fold_4_lbl)
        
        validation_fold = fold_5
        validation_fold_lbls= fold_5_lbl
    
    training_folds=np.array(training_folds)
    training_folds_labels=np.array(training_folds_labels)
    return training_folds , training_folds_labels,  validation_fold, validation_fold_lbls
   # print(validation_fold_lbls.dtype)
    
'''
 validation_fold=validation_fold.tolist()
 validation_fold_lbls=validation_fold_lbls.tolist()
 #training_folds = training_folds.tolist()
 #training_folds_labels=training_folds_labels.tolist()
 '''
   
    
'''
validation_fold=np.array(validation_fold)

validation_fold_lbls=np.array(validation_fold_lbls)
'''    
classifier = NN.NearestNeighbor()

accuracies = []
avg_accuracies = []
std = []
rng = list(range(1,11)) + np.arange(20, 101, 10).tolist()
for k in rng:
    k_accuracies = []
    print("for k = " + str(k) + ":")
    for i in range(1,6):
        training_folds , training_folds_labels,  validation_fold, validation_fold_lbls= generate_validation_fold (fold_1,fold_2,fold_3,fold_4,fold_5,fold_1_lbl,fold_2_lbl,fold_3_lbl,fold_4_lbl,fold_5_lbl, i)   
        
        training_folds = np.reshape(training_folds, (training_folds.shape[0], -1))
        validation_fold = np.reshape(validation_fold, (validation_fold.shape[0], -1))
        classifier.train(training_folds,training_folds_labels)
        validate_predict= classifier.predict(validation_fold,k,'L2')
        #print(validate_predict)
        num_test = validation_fold.shape[0]
        #print(num_test)
        num_correct = np.sum(validate_predict == validation_fold_lbls)
        accuracy = float(num_correct) / num_test
        k_accuracies.append(accuracy)
        
        
        print("for validation fold " + str(i) + ":")
        print ('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
    accuracies.append(k_accuracies)

#print(training_folds.shape)
#print(validation_fold.shape)

'''
#For all data ACCR
# generating the ACCR for the testing data for k = 60
training_data = np.reshape(training_data, (training_data.shape[0], -1))
testing_data = np.reshape(testing_data, (testing_data.shape[0], -1))
classifier.train(training_data,training_labels)
test_predict= classifier.predict(testing_data,60,'L2')
num_test = testing_data.shape[0]
num_correct = np.sum(test_predict == testing_labels)
accuracy = float(num_correct) / num_test


print("for testing data" )
print ('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
'''

'''
#For all data ACCR (greyscale)
# generating the ACCR for the testing data (greyscale) for k = 60
rgb_weights = [0.2989, 0.5870, 0.1140]
training_data = np.dot(training_data[...,:3], rgb_weights )
testing_data = np.dot(testing_data[...,:3], rgb_weights )
training_data = np.reshape(training_data, (training_data.shape[0], -1))
testing_data = np.reshape(testing_data, (testing_data.shape[0], -1))
classifier.train(training_data,training_labels)
test_predict= classifier.predict(testing_data,60,'L2')
num_test = testing_data.shape[0]
num_correct = np.sum(test_predict == testing_labels)
accuracy = float(num_correct) / num_test


print("for testing data(greyscale) and calculating ACCR:" )
print ('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

'''

'''
# generating the CCRn for the testing data for k = 60
training_data = np.reshape(training_data, (training_data.shape[0], -1))
daisy = np.reshape(daisy, (daisy.shape[0], -1))
dandelion = np.reshape(dandelion, (dandelion.shape[0], -1))
roses = np.reshape(roses, (roses.shape[0], -1))
sunflowers = np.reshape(sunflowers, (sunflowers.shape[0], -1))
tulips = np.reshape(tulips, (tulips.shape[0], -1))

classifier.train(training_data,training_labels)
daisy_predict= classifier.predict(daisy,60,'L2')
dandelion_predict= classifier.predict(dandelion,60,'L2')
roses_predict= classifier.predict(roses,60,'L2')
sunflowers_predict= classifier.predict(sunflowers,60,'L2')
tulips_predict= classifier.predict(tulips,60,'L2')
num_test = 100
daisey_num_correct = np.sum(daisy_predict == daisy_lbls)
dandelion_num_correct = np.sum(dandelion_predict == dandelion_lbls)
roses_num_correct = np.sum(roses_predict == roses_lbls)
sunflowers_num_correct = np.sum(sunflowers_predict == sunflowers_lbls)
tulips_num_correct = np.sum(tulips_predict == tulips_lbls)
daisy_accuracy = float(daisey_num_correct) / num_test
dandelion_accuracy = float(dandelion_num_correct) / num_test
roses_accuracy = float(roses_num_correct) / num_test
sunflowers_accuracy = float(sunflowers_num_correct) / num_test
tulips_accuracy = float(tulips_num_correct) / num_test



print("for each class to calculate the CCRN:" )
print ('daisey accuracy: Got %d / %d correct => accuracy: %f' % (daisey_num_correct, num_test, daisy_accuracy))
print ('dandelion accuracy: Got %d / %d correct => accuracy: %f' % (dandelion_num_correct, num_test, dandelion_accuracy))
print ('roses accuracy: Got %d / %d correct => accuracy: %f' % (roses_num_correct, num_test, roses_accuracy))
print ('sunflowers accuracy: Got %d / %d correct => accuracy: %f' % (sunflowers_num_correct, num_test, sunflowers_accuracy))
print ('tulips accuracy: Got %d / %d correct => accuracy: %f' % (tulips_num_correct, num_test, tulips_accuracy))
'''

avg_accuracies=np.average(accuracies,axis=1)
std = np.std(accuracies, axis = 1)

x = rng
#print (x)
y = accuracies
#print(y)
accuracies_dict = dict(zip(x,accuracies))
#print(accuracies_dict[1])

plt.pyplot.plot(x , avg_accuracies , alpha = 1)
for k in accuracies_dict:
    #print(k)
    c = np.random.random_sample(3)
    for i in accuracies_dict[k]:
        #print("i"+ str(i))
        plt.pyplot.scatter(k , i , color = c , alpha = 1)
     

plt.pyplot.title("5 fold Cross-validation Results Vs K")
plt.pyplot.xlabel("K Values")
plt.pyplot.ylabel("Cross Validation Results")
plt.pyplot.figure()
plt.pyplot.show()



plt.pyplot.plot(x , std , color = "orange" , marker='^')
plt.pyplot.title("Kfolds Standard Deviation vs K Values")
plt.pyplot.xlabel("K Values")
plt.pyplot.ylabel("Folds Standard Deviation")
plt.pyplot.figure()
plt.pyplot.show()
#print(avg_accuracies)
#print(std)
#print(accuracies)


'''
print(fold_1.shape)
plt.pyplot.imshow(fold_1[600])
print(fold_1_lbl.shape)
print(fold_1_lbl[600])
'''
'''
print (testing_labels[499])
plt.pyplot.figure()
plt.pyplot.imshow(testing_data[499])
print (training_labels[499])
plt.pyplot.figure()
plt.pyplot.imshow(training_data[499])
'''

#print (training_labels)
#print (testing_data[0][0].shape)
#plt.pyplot.imshow(training_data[1])
#print (training_data.shape)

