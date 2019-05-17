# Genre music classification

### Classification of the music tracks to different music genres.

#### Description

The purpose of this application is to train and test the model of classifier of music based on its genre.
Preparing and downloading dataset are automatically executed by the algorithm.

To start training, run this commands:
```
pip install -r requirements.txt
python train.py
```

After model is created, to test model run command:
```
python test.py
```

To create confusion matrix:
```
cd utils
python plot_matrix.py
```
Confusion matrix can be found under `conf_matrix.png`.

#### Database:

GTZAN - 1000 tracks - 30 seconds - 10 genres, 100 tracks each

#### Links:

Database - http://opihi.cs.uvic.ca/sound/genres.tar.gz
