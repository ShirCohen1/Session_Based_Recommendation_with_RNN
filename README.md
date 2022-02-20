# Session-based Recommendation using Deep Learning (RNN)
- This project is part of RecSys course includes PyTorch Implementation of the GRU4REC model and improvments for this model.
- Original paper: [Session-based Recommendations with Recurrent Neural Networks(ICLR 2016)](https://arxiv.org/pdf/1511.06939.pdf)
- Extension over the Original paper: [Recurrent Neural Networks with Top-k Gains for Session-based
Recommendations(CIKM 2018)](https://arxiv.org/abs/1706.03847)
- This code is based on [pyGRU4REC](https://github.com/yhs-968/pyGRU4REC) that is implemented by Younghun Song (yhs-968) and [original Theano code written by the authors of the GRU4REC paper](https://github.com/hidasib/GRU4Rec)
- This Version supports TOP1, BPR, TOP1-max, BPR-max, and Cross-Entropy Losses.

## Usage

### Dataset
RecSys Challenge 2015 Dataset can be retreived from [HERE](https://www.kaggle.com/chadgostopp/recsys-challenge-2015)

### Run the model
- You need to run [Main_RecSys_Project.ipynb](https://github.com/ShirCohen1/Session_Based_Recommendation/blob/main/Main_RecSys_Project.ipynb) to obtain training data and testing data. In the paper, only the training set was used, the testing set is ignored.
- The training set itself is divided into training and testing where the testing split is the last day sessions.

The format of data is similar to that obtained from RecSys Challenge 2015:
- Filenames
    - Training set should be named as `recSys15TrainOnly.txt`
    - Test set should be named as `recSys15Valid.txt`
    
    - Sample Training set should be named as `train_sample.txt`
    - Sample Test set should be named as `test_sample.txt`
- Contents
    - `recSys15TrainOnly.txt`, `recSys15Valid.txt` should be the tsv files that stores the pandas dataframes that satisfy the following requirements:
        - The 1st column of the file should be the integer Session IDs with header name SessionID
        - The 2nd column of the file should be the Timestamps with header name Time 
        - The 3rd column of the file should be the integer Item IDs with header name ItemID
       
     - `train_sample.txt`, `test_sample.txt` should be the tsv files that stores the pandas dataframes that satisfy the following requirements:
         - The 1st column of the file should be the integer Session IDs with header name SessionID
        - The 2nd column of the file should be the Timestamps with header name Time 
        - The 3rd column of the file should be the integer Item IDs with header name ItemID
        - The 4th column of the file should be the integer with header name time_spent
        



## Results

you should check results folder in this repo

