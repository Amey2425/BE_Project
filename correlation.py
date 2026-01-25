import pandas as pd

dataset = pd.read_csv("./dataset/health_data.csv")
corr_matrix = dataset.corr()
cardio_corr = corr_matrix['cardio']
print(cardio_corr)

'''
                cardio
age            0.238159
gender         0.008109
height        -0.010821
weight         0.181660
ap_hi          0.054475
ap_lo          0.065719
cholesterol    0.221147
gluc           0.089307
smoke         -0.015486
alco          -0.007330
active        -0.035653
cardio         1.000000

'''
dataset = pd.read_csv("./dataset/cleaned_health_data.csv")
corr_matrix = dataset.corr()
cardio_corr = corr_matrix['cardio']
print(cardio_corr)

'''
age            0.239403
gender         0.006644
height        -0.012806
weight         0.180762
ap_hi          0.430811
ap_lo          0.343394
cholesterol    0.221099
gluc           0.089272
smoke         -0.016233
alco          -0.008192
active        -0.038301
cardio         1.000000
'''

dataset_location = "./dataset/final_health_data.csv"
dataset = pd.read_csv(dataset_location)
corr_matrix = dataset.corr()
cardio_corr = corr_matrix['cardio']
print(dataset_location)
print(cardio_corr)
print("\n-------------\n")

'''
               cardio
age            0.239403
ap_hi          0.430811
ap_lo          0.343394
cholesterol    0.221099
cardio         1.000000
'''

dataset_location = "./dataset/global/test.csv"
dataset = pd.read_csv(dataset_location)
corr_matrix = dataset.corr()
cardio_corr = corr_matrix['cardio']
print(dataset_location)
print(cardio_corr)
print("\n-------------\n")


for i in range(1,4):
    dataset_location = f"./dataset/client{i}/training/train.csv"
    dataset = pd.read_csv(dataset_location)
    corr_matrix = dataset.corr()
    cardio_corr = corr_matrix['cardio']
    print(dataset_location)
    print(cardio_corr)
    print("\n-------------\n")

    dataset_location = f"./dataset/client{i}/testing/test.csv"
    dataset = pd.read_csv(dataset_location)
    corr_matrix = dataset.corr()
    cardio_corr = corr_matrix['cardio']
    print(dataset_location)
    print(cardio_corr)
    print("\n-------------\n")

'''
./dataset/final_health_data.csv
age            0.239403
ap_hi          0.430811
ap_lo          0.343394
cholesterol    0.221099
cardio         1.000000
Name: cardio, dtype: float64

-------------

./dataset/global/test.csv
age            0.237525
ap_hi          0.437570
ap_lo          0.338886
cholesterol    0.216824
cardio         1.000000
Name: cardio, dtype: float64

-------------

./dataset/client1/training/train.csv
age            0.237395
ap_hi          0.435163
ap_lo          0.344378
cholesterol    0.221905
cardio         1.000000
Name: cardio, dtype: float64

-------------

./dataset/client1/testing/test.csv
age            0.230377
ap_hi          0.411851
ap_lo          0.321211
cholesterol    0.232448
cardio         1.000000
Name: cardio, dtype: float64

-------------

./dataset/client2/training/train.csv
age            0.244149
ap_hi          0.429887
ap_lo          0.345880
cholesterol    0.227908
cardio         1.000000
Name: cardio, dtype: float64

-------------

./dataset/client2/testing/test.csv
age            0.241591
ap_hi          0.425319
ap_lo          0.358698
cholesterol    0.227889
cardio         1.000000
Name: cardio, dtype: float64

-------------

./dataset/client3/training/train.csv
age            0.243414
ap_hi          0.429934
ap_lo          0.346562
cholesterol    0.213604
cardio         1.000000
Name: cardio, dtype: float64

-------------

./dataset/client3/testing/test.csv
age            0.226907
ap_hi          0.420907
ap_lo          0.340547
cholesterol    0.218704
cardio         1.000000
Name: cardio, dtype: float64

-------------
'''