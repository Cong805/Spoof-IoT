# Spoof-IoT
This github repo describes how to spoof IoT device scanning activities through adversarial examples.   
This project collected about 170,000 banner data, and we selected some of them to study the generation of adversarial samples.
# Shadow Model

This part provides the following models: 
```
TextCNN, TextRNN, TextRCNN, TextRNN_Att, DPCNN.
```
#### Training
If retraining on the original data set in this project, you only need to execute:
```
python run.py --type train --model <model name> --embedding <random/ pre_trained>
```
If the data set is updated, you need to perform the following steps: 
first extract the pre-trained word vector, and then select the model for training.
```
python utils.py
```
```
python run.py --type train --model <model name> --embedding <random/ pre_trained>
```
In addition, we provide a method for training word vectors based on ```train.txt```:
```
python get_wordvector.py
```
#### Test
```
python run.py --type test --model <model name> --file <path name>
```
The path name is the file path of your test data.
# Model Adversarial Samples
This section provides a method for generating adversarial samples to spoof model-based IoT device scanning activities.
