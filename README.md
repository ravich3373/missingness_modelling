# ds_for_healthcare_project

# Documentation

## Note
1. This repo is an Educational project work.
2. This repo used code from [Computational Brain Imaging Group (CBIG)](https://github.com/ThomasYeoLab/CBIG)

## Dataset
0. You'll need permission from ADNI to access the data, which is needed to run this repo.
1. It is important to understand what data needs to be used for modelling.
2. We rely on prior work, [TADPOLE Challange](https://tadpole.grand-challenge.org/)challenge to decide on what CSV files and what features from each CSV file to be used for modelling.
3. We also plan to extend the above work by using previously unused features for modelling.


# Usage

1. Create output directory, if not exist.
```
mkdir output
```

2. Generate 20 cross-validation folds.
```
python3 gen_cv_fold.py \
    --spreadsheet data/TADPOLE_D1_D2.csv \
    --features data/features \
    --folds 20 \
    --outdir output

or 

python3 gen_cv_fold.py --spreadsheet data/TADPOLE_D1_D2.csv --features data/features --folds 20 --outdir output
```

3. Create training and test data, model filling

```
python3  gen_cv_pickle.py \
    --spreadsheet data/TADPOLE_D1_D2.csv \
    --features data/features \
    --mask output/fold0_mask.csv \
    --strategy model \
    --batch_size 128 \
    --out output/test.f0.pkl

python3 gen_cv_pickle.py \
    --spreadsheet data/TADPOLE_D1_D2.csv \
    --features data/features \
    --mask output/fold1_mask.csv \
    --strategy model \
    --batch_size 128 \
    --out output/test.f1.pkl

or 

python3 gen_cv_pickle.py --spreadsheet data/TADPOLE_D1_D2.csv --features data/features --mask output/fold0_mask.csv --strategy model --batch_size 128 --out output/test.f0.pkl

```

4. Train MinimalRNN model, first fold

```
python3 train.py --verbose \
    --data output/test.f0.pkl \
    --i_drop 0.1 \
    --h_drop 0.4 \
    --h_size 128 \
    --nb_layers 2 \
    --epochs 100 --lr 0.001333218 --model MinRNN --weight_decay 1e-7 \
    --out output/model.f0.pt

or 

python3 train_transformer.py --verbose --data output/test.f0.pkl --i_drop 0.1 --h_drop 0.4 --h_size 128 --nb_layers 2 --epochs 100 --lr 0.001290666 --model MinRNN --weight_decay 1e-5 --out output/model.f0.pt
```

5. Train MinimalRNN model, second fold
```
python3 -m train --verbose \
    --data output/test.f1.pkl \
    --i_drop 0.1 \
    --h_drop 0.4 \
    --h_size 128 \
    --nb_layers 2 \
    --epochs 100 --lr 0.001333218 --model MinRNN --weight_decay 1e-7 \
    --out output/model.f1.pt
```

6. Apply trained model on test set
```
python -m predict --checkpoint output/model.f0.pt --data output/test.f0.pkl \
    -o output/prediction_test.f0.csv
python -m predict --checkpoint output/model.f1.pt --data output/test.f1.pkl \
    -o output/prediction_test.f1.csv

or 
python3 predict_transformer.py --checkpoint output/model.f0.pt --data output/test.f0.pkl -o output/prediction_test.f0.csv
```

7. Evaluate prediction on test set, first fold
```
python3 -m evaluation --reference output/fold0_test.csv --prediction output/prediction_test.f0.csv
python3 -m evaluation --reference output/fold1_test.csv --prediction output/prediction_test.f1.csv

or 

python3 evaluation.py --reference output/fold0_test.csv --prediction output/prediction_test.f0.csv
```