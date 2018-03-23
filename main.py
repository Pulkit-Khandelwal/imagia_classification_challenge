"""
Run this main.py file to see a demo!
Read the README.md file first
"""


from classifier.load_data import *
from classifier.models import Models
from classifier.train import *


print('TRAINING DATA PREP')
filename_list, labels_list, annotation_list = load_matfile('data/lists/train_list.mat')
data_frame = create_dataframe(filename_list, labels_list, annotation_list, 'data/annotation/')
pickle_file(data_frame=data_frame, file_to_save='train_data.pickle')

x_train = to_numpy_array(data_frame, image_shape=(224, 224), data_path='data/images/')
y_train = labels_to_logical(labels_list)

print('COMPILE and TRAIN MODEL')
model = Models()
model = model.TransferFine(top_layers=True)
history, model = train(model, x_train, y_train, split=0.8, early_stopping=True, epochs=5)

print('Saving: Model Architecture and Weights')
model.save('save_architecture.h5')
model.save_weights('save_model_weights.h5')

print('TEST DATA PREP')
filename_list, labels_list, annotation_list = load_matfile('data/lists/test_list.mat')
data_frame = create_dataframe(filename_list, labels_list, annotation_list, 'data/annotation/')
pickle_file(data_frame=data_frame, file_to_save='test_data.pickle')

x_test = to_numpy_array(data_frame, image_shape=(224, 224), data_path='data/images/')
y_test = labels_to_logical(labels_list)

print('PREDICTING CLASSES')
with open(r"data/labels_dict.pickle", "rb") as input_file:
    dictionary_labels_classes = pickle.load(input_file)
    
classes = []
class_dogs = []
for i in range(len(x_test)):
    pred = np.argmax(model.predict(np.expand_dims(resize(x_test[i], (299, 299)), axis=0)))+1
    classes.append(pred)
    class_dogs.append(dictionary_labels_classes[pred])

print('Predicted Labels')
print(classes)
print(class_dogs)

print('Let us see some training and validation curves')
# list all data in history

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plot a test image
io.imshow(x_test[0])
plt.show()

