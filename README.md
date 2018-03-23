Architectural choices:

1. Pre-trained models:
    Resnet50
    InceptionNet
    Xception
    Vgg16
    
2. Transfer learning and Fine Tuning on Xception Net
 
3. Make your own ConvNet model from scratch


I have used state of the art classification architectures. But, have used transfer learning and fine tuning as the crux of the
model. The given data is part of imagenet. Architectures 1-4 have been trained on imagenet and the weights are easily available on Keras of any other framework. Transfer learning is used to initilaize weights of the network and serves as the base model. Then using Fine Tuning layers can be refined and trained by freezing some other part of the network. Usually the top layers are fine tuned not we should make sure that the weights of the lower layers are not perturbed. Else this might cause the network to to go crazy with gradient mismatch.

I have also evaluated results on some pre-trained network directly without any form of tuning. Lastly for comparison, one can also train the network from scratch. Though this is not recommended if one has a network which has been trained on similar network. Each image has been cropped and the entire train dataset is normalized with zero mean and unit variance. I have not done any kind of data augmentation because the data is already large enough for our purposes.

I have created nice pandas dataframes which you might find in the code. I have made an interface to bind all the code together. I focused more on neatness of the code and network choices. Due to constraint of resources I could not train the network on the entire dataset for the Transfer Leanring section; and have only tested on a small section. But, **main.py** runs it for you on the entire dataset both for train and test dataset. If you want, I can set up an account on digitalocean and train it there.


[1] Download the data from the drive.
Here is the link: https://drive.google.com/file/d/1IbHpqU9KnHiuitOTz4m8YXtCot2a7RgN/view?usp=sharing

Extract the file and yo should get a folder named data. Now, download the project from gitHub and extract it. Copy the "data" folder to this folder.


[2] The data should be organized as follow:

     /code/

     /data/

          /images

          /annotation

          /lists

               /file_list.mat

               /train_list.mat

               /test_list.mat

          labels_dict.pickle
     


[3] labels_dict.pickle is dictionary which maps the integer labels 1,2,3,.....,120 to their corresponding breed of dogs.

[4] The code is written in the form of a nice interface with all docstrings in place for every method, class and module. Run main.py file to train and test how Transfer Learning and Fine Tuning works.

We will Transfer Learn and Fine Tune the xCeption network. Also, refer to the Jupyter Notebook for more examples.


[5] Note:
1. Here, in main.py we will train and test on the provided entire train and test datasets
But....

2. For quick testing of whether the code works or not
you can train on 200 examples and test on 10 as I have to show how the different code components works in the Jupyter Notebook.


[6] Requirements:
pip3 install -r requirements.txt
