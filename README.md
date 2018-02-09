# MSc_Onlab

## train:
```
python3 train_uj.py <input data file> <experiment number> '<morph. tag separator>' <source morph tags place in line> <target morph tags place in line>
```

  Ha forrás vagy cél morfológiai tag nincs az adott bemeneten, akkor helyette -1 írandó.
  ```
  pl. python3 train_uj.py task1.tsv 1 ';' 2 -1
  ```
  
## test:
```
python3 test_uj.py <input data file> <trained model name> '<morph tag separator>' <source morph tags place> <target morph tags place>
```

    ```
    pl. python3 test_uj.py task1_test.tsv trained_model0 ';' 2 -1
    ```
## inference:

```
python3 inference_uj.py "<word><tab><moprh tags separated by ;>" <trained model name>
```

    ```
    pl. python3 inference_uj.py "államigazgatás   N;IN+ESS;SG" trained_model0
    ```
