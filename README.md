# MSc_Onlab

train:
```
python3 train_uj.py <input data file> <experiment number> '<morph. tag separator>' <source morph tags place in line> <target morph tags place in line>
```  
  Ha forrás vagy cél morfológiai tag nincs az adott bemeneten, akkor helyette -1 írandó.
  
test:
python3 test_uj.py <input data file> <trained model name> '<morph tag separator>' <source morph tags place> <target morph tags place>
  
inference:
python3 inference_uj.py "<word><tab><moprh tags separated by ;>" <trained model name>
