

[1] Download the data from the drive.
Here is the link. Extract the file and
You should get a folder named data.
Now, download the project from gitHub and extract it. Copy the "data" folder to this folder.


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


[3] labels_dict.pickle is dictionary which maps the integer labels 1,2,3,.....120 to their corresponding breed of dogs.

[4] Run main.py file to train and test
how Transfer Learning and Fine Tuning works.
We will Transfer Learn and Fine Tune the XCeption network.
Also, refer to the Jupyter Notebook for more examples.

[5] Note:
1. Here, in main.py we will train and test on the provided entire train and test datasets
But....
2. For quick testing of whether the code works or not
you can train on 200 examples and test on 10 as I have to show how the different code components works in the Jupyter Notebook.

[6] Requirements:
Bumpy, script, sklearn, skim age, h5py, keras and tensorflow




‌