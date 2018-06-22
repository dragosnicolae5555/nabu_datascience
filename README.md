# nabu_datascience

### Requirements

To satisfy requirments please create new conda environment
* make sure you have conda installed
* go inside the repo folder
* run `conda env update` - it will create the `nabu` environment
 
### Running
To run the script on the video follow the steps:
* change the video path and other parameters inside `defaults.py`
* change the conda env `source activate nabu`
* run the code `python run.py`

### Notes
* the pyannote package was copied to the repo to allow modyfing the function and classes
* sortedcollections pyckage was copied becouse of the bug in the official repo (update method of ValueSortedDict was corrupted)
* the code was tested on ubuntu