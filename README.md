# Composing Music With Recurrent Neural Networks

This repo is a replication work of [**Biaxial RNN music composition**](https://www.danieldjohnson.com/2015/08/03/composing-music-with-recurrent-neural-networks/) by Daniel D. Johnson in 2015. In our work, we used tensorflow and keras to rebuild his model, and achieve the goal to generate classical pieces. The report can be found [here](https://github.com/ChengweiWang3210/Music_generator_LSTM/blob/main/E4040.2020Fall.QMSS.report.yl4318.yl4319.cw3210.pdf). 

**Note**: we stored our trained model weitghts in google drive, as it is too large for github repo. Here is the [link](https://drive.google.com/drive/folders/1N-pdq5LiT7ppJ2DnkrYrtmAZT4fX7J4s?usp=sharing). Thank you! 

### 1. How to run the code

The major part of our work -- the model itself can be found in the notebook file: [trainmodel.ipynb](https://github.com/ecbme4040/e4040-2020Fall-Project-QMSS-yl4318-yl4319-cw3210/blob/main/trainmodel.ipynb). 

The model has already been constructed in the identical structure as the original work, with 2 layers of LSTM for time axis (each layer has 300 hidden units) and 2 layers of LSTM for note axis (respectively have 100 and 50 hidden units in these 2 layers.)

Also the input and output data has been taken care of in a seperate python file and returned a generator waiting for the model fitting process. 

Therefore, after running all the code chucks preceding the below one, the model is prepared to been trained. 

The 4th code chuck in the notebook, shown below, assumes the work of training the weight matrices and test the model effectiveness on validation data set. 

```python
history_LSTM = model.fit(train_ds, validation_data=val_ds, 
                         steps_per_epoch=30,
                         validation_steps=5, epochs=30)
```

The code chunk underneath this chuck are working to save, or reload the saved model weights, visualizing the key metrics within epochs, and predict (or generate) a brand new song based on the input data. 

### 2. Key functions of each files

Our model is stored in the major jupyter notebook file named [trainmodel.ipynb](https://github.com/ecbme4040/e4040-2020Fall-Project-QMSS-yl4318-yl4319-cw3210/blob/main/trainmodel.ipynb). 

We also has two python file assisting the data transformation and customizing neural network layers. 

#### 2.1 Data.py

In the [data.py](https://github.com/ecbme4040/e4040-2020Fall-Project-QMSS-yl4318-yl4319-cw3210/blob/main/data.py) file, we built three functions here.

The first one is ```toArray()```, which is used to read in the MIDI files and transform the messages and events in each track into a more structured matrix, so that we could later construct input data and output data based on this matrix. It take one argument ```file```, the name and path of the MIDI file, and return an numpy array object with shape of (num_of_bear, range_of_notes=128, 2), the third dimension of 2 here means two boolean here: one indicates if the note is played, and the other indicates if the note is articulated. 

The second function in this file ```getBatch()``` is used to generate batches, which will be later fed into the model for training or validation. This function take the a string indicating the path where the data is stored. It will first of all call the previous function ```toArray``` to transform the songs in the given path to matrix and then if the matrix is valid, it will randomly choose a 8-measure long subset of the original matrix for 10 (batch_width) times. As our model actually takes the same input and output, we return the results twice in a tuple with each has the shape of (batch_width=10, num_timesteps=128, num_notes=128, 2). 

The third function in the file is ```toMidi()```, this is the function used for transforming the matrix (numpy array) predicted by the trained model back into the midi format, and store the newly generated song into local directory, so that we could enjoy the music conposed by our model. 

#### 2.2 Layers.py

In this python file, we used three custom layers to reconstruct or reshape the data when it flow throught the sequence of layers in the model with untrainable weights, and one customized loss to feed into model optimizing process to be minimized.

The first class is ```transformLayer```, this class's major method ```gen_input()``` takes the batch of input data, and translates it into a "real" input data by concatenating information like position, pitchlass, previous vicinity, previous context, and beat, which are extracted from the input data from the generator from ```genBatch()```, and returns a flattened tensor with shape of (batch_size*num_notes=10\*128, num_timesteps=128, 80). 

The second class ```changeAxisLayer1``` is used to exchange the time axis and note axis between the time-axis LSTM layers and note-axis LSTM layers in the middle of our model. In the major method ```changeAxis()``` is taking the results from the second LSTM layer for the time-axis part with hidden units as 300. The input of this function is (batch_size*num_notes = 10\*128, num_timesteps=128, 300), and we first of all reshape it to separate the first dimension into batch size and number of notes, and get shape (batch_size, num_notes, num_timestep, 300), then we transpose the tensor to exchange the second and the third dimension, like (batch_size, num_timestep, num_notes, 300). Lastly, we combined the first and the second dimension again to feed back to the note-axis LSTM layers. 

The third class ```changeAxisLayer2``` divorces the previously combined first and second dimension to feed the last layer -- Dense.  

The fourth class ```LossAligned``` truncates the first time step of our ground truth, so that the generated notes can be compared to one time step behind. To make the predicted y and the real target of the same length, we also cut off the last time step of the predicted notes. Then we calculate binary crossentropy based on these realigned sequence. 

### 3. Location of the datasets

We download the songs from website [Classical Piano Midi Page](http://www.piano-midi.de/midicoll.htm) where the author got his data. Instead of all the songs, we only choose 87 pieces of songs from 4 musicians of the similar style for our model training, and downloaded these songs and another 20 pieces for validation into our local directory, and uploaded into this repo. We did this because first all this dataset is less easy to download via a few lines of code, and they are relatively small, only taking around 3 MB in total. 

The training set data is stored in the folder named ```train```, and the validation set is stored in the folder named ```validation```.

### 4. Organization of directories

```
.
├── README.md
├── data.py
├── layers.py
├── trainmodel.ipynb
├── figures
│   ├── Blank\ diagram.png
│   ├── Flowchart.png
│   ├── Flowchart1.png
│   ├── Flowchart2.png
│   ├── GCP
│   │   ├── qmss_gcp_work_example_screenshot_1.png
│   │   ├── qmss_gcp_work_example_screenshot_2.png
│   │   └── qmss_gcp_work_example_screenshot_3.png
│   ├── biaxial.png
│   ├── network.png
│   ├── transformLayer.png
│   └── transformlayer_diagram.png
├── train
│   ├── appass_1_format0.mid
│   ├── beethoven_hammerklavier_4_format0.mid
│   ├── beethoven_les_adieux_1_format0.mid
│   ├── beethoven_les_adieux_2_format0.mid
│   ├── beethoven_les_adieux_3_format0.mid
│   ├── beethoven_opus10_1_format0.mid
│   ├── beethoven_opus10_2.mid
│   ├── beethoven_opus10_3.mid
│   ├── beethoven_opus22_1_format0.mid
│   ├── beethoven_opus22_2_format0.mid
│   ├── beethoven_opus22_3_format0.mid
│   ├── beethoven_opus90_1_format0.mid
│   ├── beethoven_opus90_2_format0.mid
│   ├── br_im6_format0.mid
│   ├── br_rhap_format0.mid
│   ├── brahms_opus117_1_format0.mid
│   ├── brahms_opus117_2_format0.mid
│   ├── brahms_opus1_1_format0.mid
│   ├── brahms_opus1_2_format0.mid
│   ├── brahms_opus1_3_format0.mid
│   ├── chp_op18.mid
│   ├── chp_op31.mid
│   ├── chpn-p1.mid
│   ├── chpn-p10.mid
│   ├── chpn-p11.mid
│   ├── chpn-p12.mid
│   ├── chpn-p13.mid
│   ├── chpn-p14.mid
│   ├── chpn-p15.mid
│   ├── chpn-p16.mid
│   ├── chpn-p17.mid
│   ├── chpn-p18.mid
│   ├── chpn-p19.mid
│   ├── chpn-p2.mid
│   ├── chpn-p20.mid
│   ├── chpn-p21.mid
│   ├── chpn-p23.mid
│   ├── chpn-p24.mid
│   ├── chpn-p3.mid
│   ├── chpn-p4.mid
│   ├── chpn-p5.mid
│   ├── chpn-p6.mid
│   ├── chpn-p8.mid
│   ├── chpn-p9.mid
│   ├── chpn_op10_e01.mid
│   ├── chpn_op10_e05.mid
│   ├── chpn_op10_e12.mid
│   ├── chpn_op23.mid
│   ├── chpn_op25_e1.mid
│   ├── chpn_op25_e12.mid
│   ├── chpn_op25_e2.mid
│   ├── chpn_op25_e4.mid
│   ├── chpn_op27_1.mid
│   ├── chpn_op33_2.mid
│   ├── chpn_op33_4.mid
│   ├── chpn_op35_1.mid
│   ├── chpn_op35_2.mid
│   ├── chpn_op35_4.mid
│   ├── chpn_op53.mid
│   ├── chpn_op7_1.mid
│   ├── chpn_op7_2.mid
│   ├── mond_1_format0.mid
│   ├── mond_2_format0.mid
│   ├── mz_311_1.mid
│   ├── mz_311_2.mid
│   ├── mz_311_3.mid
│   ├── mz_330_1.mid
│   ├── mz_330_2.mid
│   ├── mz_330_3.mid
│   ├── mz_331_1.mid
│   ├── mz_331_2.mid
│   ├── mz_331_3.mid
│   ├── mz_332_1.mid
│   ├── mz_332_2.mid
│   ├── mz_332_3.mid
│   ├── mz_333_1.mid
│   ├── mz_333_3.mid
│   ├── mz_545_2.mid
│   ├── mz_545_3.mid
│   ├── mz_570_1.mid
│   ├── mz_570_2.mid
│   ├── mz_570_3.mid
│   ├── pathetique_1.mid
│   ├── pathetique_2.mid
│   ├── pathetique_3.mid
│   ├── waldstein_1_format0.mid
│   └── waldstein_2_format0.mid
└── validation
    ├── appass_2_format0.mid
    ├── appass_3_format0.mid
    ├── beethoven_hammerklavier_1_format0.mid
    ├── beethoven_hammerklavier_2_format0.mid
    ├── beethoven_hammerklavier_3_format0.mid
    ├── beethoven_opus22_4_format0.mid
    ├── br_im2_format0.mid
    ├── br_im5_format0.mid
    ├── brahms_opus1_4_format0.mid
    ├── chpn-p22.mid
    ├── chpn-p7.mid
    ├── chpn_op25_e11.mid
    ├── chpn_op25_e3.mid
    ├── chpn_op27_2.mid
    ├── chpn_op35_3.mid
    ├── chpn_op66.mid
    ├── mond_3_format0.mid
    ├── mz_333_2.mid
    ├── mz_545_1.mid
    └── waldstein_3_format0.mid

4 directories, 122 files
```



