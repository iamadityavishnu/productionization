# Image Classifier WebApp

To train the model, 2 more directories are required: `data` and `log`.

`data` has the training data stored. While training, the data will be
downloaded to the folder. During subsequent training iterations, data
will not be downloaded again.

`log` contains tensorboard logs for the training loops. To run tensorboard,
run the command: `tensorboard --logdir=./log`.

To run the webapp, launch the command prompt and go back a directory.
Run the command: 
```
uvicorn image_classification.main:app --reload
```