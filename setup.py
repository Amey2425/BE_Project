# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os
# import numpy as np
# from dotenv import load_dotenv
# load_dotenv()
# import matplotlib.pyplot as plt

# '''
# ###################################################################
#     DATASET CREATION PHASE
    
#     Objectives:
#         1. Create a global testing dataset
#         2. Create a training and testing dataset for each client
#         3. Make it uneven to resemble real world scenario
#         4. Plot the graphs in respective folders
# ###################################################################
# '''

# def plot_distributions(data, folder_path):
#     plt.figure(figsize=(12, 10))  # Adjusted figure size
    
#     # Age distribution (years)
#     plt.subplot(2, 2, 1)  # Changed to 2x2 grid
#     plt.hist(data['age_years'], bins=20, color='blue', alpha=0.7)
#     plt.title('Age Distribution')
#     plt.xlabel('Age (years)')
    
#     # Systolic blood pressure (ap_hi) distribution
#     plt.subplot(2, 2, 2)  # Next subplot in the 2x2 grid
#     plt.hist(data['ap_hi'], bins=20, color='cyan', alpha=0.7)
#     plt.title('Systolic Blood Pressure Distribution')
#     plt.xlabel('Systolic BP (ap_hi)')
    
#     # Diastolic blood pressure (ap_lo) distribution
#     plt.subplot(2, 2, 3)
#     plt.hist(data['ap_lo'], bins=20, color='pink', alpha=0.7)
#     plt.title('Diastolic Blood Pressure Distribution')
#     plt.xlabel('Diastolic BP (ap_lo)')
    
#     # Cardiovascular disease (cardio) distribution
#     plt.subplot(2, 2, 4)
#     plt.bar(data['cardio'].value_counts().index, data['cardio'].value_counts().values, color='brown', alpha=0.7)
#     plt.title('Cardiovascular Disease Distribution')
#     plt.xlabel('Cardio (0=Absence, 1=Presence)')
    
#     # Adjust layout and save plot
#     plt.tight_layout()
#     plt.savefig(f'{folder_path}/distributions.png')
#     plt.close()


# def split_save_and_plot(client_data, client_name):
#     # Split data into 80% training and 20% testing
#     train, test = train_test_split(client_data, test_size=0.2, random_state=42)
    
#     # Save training and testing data
#     train.to_csv(f'./dataset/{client_name}/training/train.csv', index=False)
#     test.to_csv(f'./dataset/{client_name}/testing/test.csv', index=False)
    
#     print(client_name)
#     print(f"Training : {len(train)}")
#     print(f"Testing : {len(test)}")

#     # Plot and save distributions for training data
#     plot_distributions(train, f'./dataset/{client_name}/training/plots')
    
#     # Plot and save distributions for testing data
#     plot_distributions(test, f'./dataset/{client_name}/testing/plots')

#     train = train.drop(columns=['age_years'])
#     test = test.drop(columns=['age_years'])
    
#     # Save training and testing data
#     train.to_csv(f'./dataset/{client_name}/training/train.csv', index=False)
#     test.to_csv(f'./dataset/{client_name}/testing/test.csv', index=False)

# clients = ['client1', 'client2', 'client3']
# folders = ['training', 'testing']
# for client in clients:
#     for folder in folders:
#         os.makedirs(f'./dataset/{client}/{folder}/plots', exist_ok=True)

# os.makedirs(f'./dataset/global/plots', exist_ok=True)

# dataset = pd.read_csv('./dataset/final_health_data.csv')
# dataset['age_years'] = dataset['age'] / 365

# dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
# global_train, global_test = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['cardio'])
# global_test.to_csv('./dataset/global/test.csv', index=False)
# plot_distributions(global_test,'./dataset/global/plots')
# global_test = global_test.drop(columns=['age_years'])
# global_test.to_csv('./dataset/global/test.csv', index=False)

# client1_data, client2_data, client3_data = np.array_split(global_train, 3)

# # cardio_0 = global_train[global_train['cardio'] == 0]  # cardio = 0 (absence of disease)
# # cardio_1 = global_train[global_train['cardio'] == 1]  # cardio = 1 (presence of disease)


# # # Client 1: Skewed towards cardio = 0 (80% cardio = 0, 20% cardio = 1)
# # client1_cardio_0 = cardio_0.sample(frac=0.8, random_state=42)
# # client1_cardio_1 = cardio_1.sample(frac=0.2, random_state=42)
# # client1_data = pd.concat([client1_cardio_0, client1_cardio_1]).sample(frac=1, random_state=42)

# # # Client 2: Skewed towards cardio = 1 (40% cardio = 0, 60% cardio = 1)
# # client2_cardio_0 = cardio_0.drop(client1_cardio_0.index).sample(frac=0.4, random_state=42)
# # client2_cardio_1 = cardio_1.drop(client1_cardio_1.index).sample(frac=0.6, random_state=42)
# # client2_data = pd.concat([client2_cardio_0, client2_cardio_1]).sample(frac=1, random_state=42)

# # # Client 3: Skewed towards cardio = 1 (30% cardio = 0, 70% cardio = 1)
# # client3_cardio_0 = cardio_0.drop(client1_cardio_0.index).drop(client2_cardio_0.index)
# # client3_cardio_1 = cardio_1.drop(client1_cardio_1.index).drop(client2_cardio_1.index)
# # client3_data = pd.concat([client3_cardio_0.sample(frac=0.3, random_state=42), 
# #                           client3_cardio_1.sample(frac=0.7, random_state=42)]).sample(frac=1, random_state=42)

# clients_data = [client1_data, client2_data, client3_data]

# # Iterate over each client's data
# for i, data in enumerate(clients_data):
#     split_save_and_plot(data, clients[i])
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt

'''
###################################################################
    DATASET CREATION PHASE
    
    Objectives:
        1. Create a global testing dataset
        2. Create a training and testing dataset for each client
        3. Make it uneven to resemble real-world scenario
        4. Plot the graphs in respective folders
###################################################################
'''


# -------------------------------------------------------------
# PLOT DISTRIBUTION FUNCTION
# -------------------------------------------------------------
def plot_distributions(data, folder_path):
    plt.figure(figsize=(12, 10))
    
    # Age distribution
    plt.subplot(2, 2, 1)
    plt.hist(data['age_years'], bins=20, color='blue', alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age (years)')

    # Systolic BP distribution
    plt.subplot(2, 2, 2)
    plt.hist(data['ap_hi'], bins=20, color='cyan', alpha=0.7)
    plt.title('Systolic Blood Pressure')

    # Diastolic BP distribution
    plt.subplot(2, 2, 3)
    plt.hist(data['ap_lo'], bins=20, color='pink', alpha=0.7)
    plt.title('Diastolic Blood Pressure')

    # Cardio class distribution
    plt.subplot(2, 2, 4)
    plt.bar(
        data['cardio'].value_counts().index,
        data['cardio'].value_counts().values,
        color='brown',
        alpha=0.7
    )
    plt.title('Cardio Distribution (0/1)')

    plt.tight_layout()
    plt.savefig(f'{folder_path}/distributions.png')
    plt.close()


# -------------------------------------------------------------
# SPLIT + SAVE + PLOT FUNCTION
# -------------------------------------------------------------
def split_save_and_plot(client_data, client_name):
    # 80-20 split with stratification
    train, test = train_test_split(
        client_data,
        test_size=0.2,
        random_state=42,
        stratify=client_data["cardio"]
    )

    # Paths
    train_path = f'./dataset/{client_name}/training'
    test_path = f'./dataset/{client_name}/testing'

    # Save FULL (before dropping derived feature)
    train.to_csv(f'{train_path}/train_full.csv', index=False)
    test.to_csv(f'{test_path}/test_full.csv', index=False)

    # Plotting
    plot_distributions(train, f'{train_path}/plots')
    plot_distributions(test, f'{test_path}/plots')

    # Drop age_years before final save
    train_final = train.drop(columns=['age_years'])
    test_final = test.drop(columns=['age_years'])

    train_final.to_csv(f'{train_path}/train.csv', index=False)
    test_final.to_csv(f'{test_path}/test.csv', index=False)

    # Stats
    print(f"\n{client_name.upper()}:")
    print("Training samples:", len(train_final))
    print("Testing samples: ", len(test_final))


# -------------------------------------------------------------
# CREATE FOLDERS
# -------------------------------------------------------------
clients = ['client1', 'client2', 'client3']
for client in clients:
    os.makedirs(f'./dataset/{client}/training/plots', exist_ok=True)
    os.makedirs(f'./dataset/{client}/testing/plots', exist_ok=True)

os.makedirs('./dataset/global/plots', exist_ok=True)


# -------------------------------------------------------------
# LOAD & PREPARE DATASET
# -------------------------------------------------------------
dataset = pd.read_csv('./dataset/final_health_data.csv')
dataset['age_years'] = dataset['age'] / 365      # Derived feature

# Shuffle dataset
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)


# -------------------------------------------------------------
# GLOBAL TEST SET (20%)
# -------------------------------------------------------------
global_train, global_test = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42,
    stratify=dataset['cardio']
)

# Plot global test before dropping
plot_distributions(global_test, './dataset/global/plots')

# Save cleaned global test
global_test_final = global_test.drop(columns=['age_years'])
global_test_final.to_csv('./dataset/global/test.csv', index=False)

print("\nGLOBAL TEST SIZE:", len(global_test_final))


# -------------------------------------------------------------
# NON-IID CLIENT SPLITS (simple uneven, as you originally used)
# -------------------------------------------------------------
client1_data, client2_data, client3_data = np.array_split(global_train, 3)

clients_data = [client1_data, client2_data, client3_data]


# -------------------------------------------------------------
# PROCESS EACH CLIENT
# -------------------------------------------------------------
for i, data in enumerate(clients_data):
    split_save_and_plot(data, clients[i])

print("\n✔ Dataset preparation completed successfully!")
