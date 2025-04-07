import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


import lightning as L
from torch.utils.data import TensorDataset, DataLoader,Dataset

import dsgp4

from dsgp4.util import initialize_tle, propagate, propagate_batch
from torch.nn.parameter import Parameter


log_records = []  # This will store log entries as dictionaries


print("Using GPU:", torch.cuda.is_available())


torch.set_float32_matmul_precision('high')  
 

class BasicLightning(L.LightningModule):

    def __init__(self,
                 layer1_neurons = 35, # these neurons are to tune
                 layer2_neurons = 35,
                 layer3_neurons = 35,
                 layer4_neurons = 35,
                 layer5_neurons = 35,
                 layer6_neurons = 35,
                 normalization_R=6958.137, #constant to normalize pos values
                 normalization_V=7.947155867983262, #constats to normalize velocity values
                 input_correction=1e-2, 
                 output_correction=0.8):

        super().__init__()

        self.fc1=nn.Linear(6, layer1_neurons)                   #learnable layer
        self.fc2=nn.Linear(layer2_neurons,layer2_neurons)          #learnable layer
        self.fc3=nn.Linear(layer3_neurons, 6)                   #learnable layer
        self.fc4=nn.Linear(6,layer4_neurons)                    #learnable layer
        self.fc5=nn.Linear(layer5_neurons, layer5_neurons)         #learnable layer
        self.fc6=nn.Linear(layer6_neurons, 6)                   #learnable layer

        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.normalization_R=normalization_R #constant
        self.normalization_V=normalization_V #constant
        self.input_correction = Parameter(input_correction*torch.ones((6,)))    #learnable layer
        self.output_correction = Parameter(output_correction*torch.ones((6,)))  #learnable layer

    def forward(self, tles, tsinces):
        if not isinstance(tles,list):
            print("dataloader is not giving a lsit in the forward pass")
        # Get the device from one of the model's parameters.
        device = next(self.parameters()).device
        print("forward pass")
        print(device)

        # Ensure tsinces is a tensor on the correct device.
        if not isinstance(tsinces, torch.Tensor):
            tsinces = torch.tensor(tsinces, dtype=torch.float32, device=device)
        else:
            tsinces = tsinces.to(device)

        ###################### SECTION 1: Initialize the batch ###########################
        is_batch = hasattr(tles, '__len__')
        if is_batch:
            # For the batch case, assume initialize_tle returns a tuple;
            # the second element contains the TLE objects.
            _, tles = initialize_tle(tles, with_grad=True)
            x0 = torch.stack((tles._ecco, tles._argpo, tles._inclo,
                            tles._mo, tles._no_kozai, tles._nodeo), dim=1)
        else:
            # Single TLE case.
            initialize_tle(tles, with_grad=True)
            x0 = torch.stack((tles._ecco, tles._argpo, tles._inclo,
                            tles._mo, tles._no_kozai, tles._nodeo), dim=0).reshape(-1, 6)
        
        # Move x0 to the same device as the model.
        x0 = x0.to(device)

        #################### SECTION 2: TLE Optimization #############################
        x = self.leaky_relu(self.fc1(x0))
        x = self.leaky_relu(self.fc2(x))
        x = x0 * (1 + self.input_correction * self.tanh(self.fc3(x)))

        # Update TLE attributes with the optimized values.
        tles._ecco    = x[:, 0]
        tles._argpo   = x[:, 1]
        tles._inclo   = x[:, 2]
        tles._mo      = x[:, 3]
        tles._no_kozai = x[:, 4]
        tles._nodeo   = x[:, 5]

        # For batch mode, update each TLE object's propagation time (_t)
        # and ensure attributes like _mdot are moved to the same device.
    

        #################### Propagation #############################
        if is_batch:
            states_teme = propagate_batch(tles, tsinces)
        else:
            states_teme = propagate(tles, tsinces)

        # Ensure the propagation output is on the correct device.
        states_teme = states_teme.to(device).reshape(-1, 6)
        
        # Normalize and form the output.
        x_out = torch.cat((states_teme[:, :3] / self.normalization_R,
                        states_teme[:, 3:] / self.normalization_V), dim=1)

        x = self.leaky_relu(self.fc4(x_out))
        x = self.leaky_relu(self.fc5(x))
        x = x_out * (1 + self.output_correction * self.tanh(self.fc6(x)))
        return x

  
    






###################### model architecture ends###################################### MODEL IS READY








###################### MODEL STEPS################################

    def training_step(self, batch, batch_idx):
        print("training step start")

        
        # Create TLE objects for the batch
        training_TLE_objects = [dsgp4.tle.TLE(tle_data, device=device) for tle_data in batch[0]]
        training_ephemeris = batch[1]
        training_prop_time = batch[2]


        # Forward pass: obtain predicted positions and velocities
        training_PV_hat = self.forward(training_TLE_objects, training_prop_time)
      

        # Normalize the ephemeris data (assuming ephemeris is arranged as [position, velocity])
        training_ephemeris[:, :3] = training_ephemeris[:, :3] / RAJ_MODEL.normalization_R
        training_ephemeris[:, 3:] = training_ephemeris[:, 3:] / RAJ_MODEL.normalization_V

    

        # Compute the difference between prediction and true ephemeris
        diff = training_PV_hat - training_ephemeris

        # Compute the overall RMSE loss
        training_loss = torch.mean(diff ** 2)
        
        # Compute MSE for position and velocity separately
        p_loss = torch.mean((training_PV_hat[:, :3] - training_ephemeris[:, :3]) ** 2)
        v_loss = torch.mean((training_PV_hat[:, 3:] - training_ephemeris[:, 3:]) ** 2)
        
        
        log_records.append({
        'epoch': self.current_epoch,
        'batch': self.global_step,
        'phase': 'train',
        'position_loss': p_loss.item(),
        'velocity_loss': v_loss.item(),
        'total_loss': training_loss.item()
})


        
        # Log the individual position and velocity losses as well as the overall MSE
        print("Position RMSE:", p_loss.item())
        print("Velocity RMSE:", v_loss.item())
        print("Overall RMSE:", training_loss.item())
        print("training step end")

        return training_loss



    def validation_step(self, batch, batch_idx):
        print("validation step start")

        # Create TLE objects for the batch
        validation_TLE_objects = [dsgp4.tle.TLE(tle_data, device=device) for tle_data in batch[0]]
        validation_ephemeris = batch[1]
        validation_prop_time = batch[2]

        # Forward pass
        validation_PV_hat = self.forward(validation_TLE_objects, validation_prop_time)

        # Normalize ground truth ephemeris
        validation_ephemeris[:, :3] = validation_ephemeris[:, :3] / RAJ_MODEL.normalization_R
        validation_ephemeris[:, 3:] = validation_ephemeris[:, 3:] / RAJ_MODEL.normalization_V

        # Compute the difference
        diff = validation_PV_hat - validation_ephemeris

        # Overall MSE loss
        validation_loss = torch.mean(diff ** 2)

        # Positional and velocity MSE
        p_loss = torch.mean((validation_PV_hat[:, :3] - validation_ephemeris[:, :3]) ** 2)
        v_loss = torch.mean((validation_PV_hat[:, 3:] - validation_ephemeris[:, 3:]) ** 2)

        log_records.append({
        'epoch': self.current_epoch,
        'batch': self.global_step,
        'phase': 'val',
        'position_loss': p_loss.item(),
        'velocity_loss': v_loss.item(),
        'total_loss': validation_loss.item()
})


        print("Position RMSE:", p_loss.item())
        print("Velocity RMSE:", v_loss.item())
        print("Overall RMSE:", validation_loss.item())
        print("validation step end")




        return validation_loss


    def test_step(self, batch, batch_idx):
        print("test step start")

        # Create TLE objects for the batch
        test_TLE_objects = [dsgp4.tle.TLE(tle_data, device=device) for tle_data in batch[0]]
        test_ephemeris = batch[1]
        test_prop_time = batch[2]

        # Forward pass
        test_PV_hat = self.forward(test_TLE_objects, test_prop_time)

        # Normalize ground truth ephemeris
        test_ephemeris[:, :3] = test_ephemeris[:, :3] / RAJ_MODEL.normalization_R
        test_ephemeris[:, 3:] = test_ephemeris[:, 3:] / RAJ_MODEL.normalization_V

        # Compute the difference
        diff = test_PV_hat - test_ephemeris

        # Overall MSE loss
        test_loss = torch.mean(diff ** 2)

        # Positional and velocity MSE
        p_loss = torch.mean((test_PV_hat[:, :3] - test_ephemeris[:, :3]) ** 2)
        v_loss = torch.mean((test_PV_hat[:, 3:] - test_ephemeris[:, 3:]) ** 2)


        log_records.append({
        'epoch': self.current_epoch,
        'batch': self.global_step,
        'phase': 'test',
        'position_loss': p_loss.item(),
        'velocity_loss': v_loss.item(),
        'total_loss': test_loss.item()
})


        print("Position RMSE:", p_loss.item())
        print("Velocity RMSE:", v_loss.item())
        print("Overall RMSE:", test_loss.item())
        print("test step end")




        return test_loss






    def configure_optimizers(self):
        return Adam(self.parameters(), lr=LEARNING_RATE)



#######################################DATA RELATED#########################################

##########HYPERPARAMETERS#############
NUM_EPOCHS = 30
#TOTAL_SAMPLES = len(TRAIN_Dataset)
#n_iterations = TOTAL_SAMPLES/100         #number of batches loaded in an epoch
LEARNING_RATE =0.003
BATCH_SIZES = 1000
########################################


##creating my data class##  ## lazy for line by line, eager for load whole csv##
class Raj_Dataset(Dataset):
    def __init__(self, data_directory, mode='lazy'):
        """
        Args:
            data_directory (str): Path to the CSV file.
            mode (str): 'lazy' for memory-efficient row-by-row reading,
                        'eager' to load the entire CSV into memory.
        """
        self.data_directory = data_directory
        self.mode = mode.lower()
        assert self.mode in ['lazy', 'eager'], "Mode must be either 'lazy' or 'eager'."

        if self.mode == 'eager':
            self.df = pd.read_csv(self.data_directory)
            self.n_samples = len(self.df)
        else:
            with open(self.data_directory, 'r') as f:
                self.header = f.readline()
                self.n_samples = sum(1 for _ in f)

    def __getitem__(self, index):
        if self.mode == 'eager':
            row = self.df.iloc[index]
            tle = row.iloc[6]
            y = row.iloc[0:6].values.astype(np.float32)
            t = float(row.iloc[7])
        else:
            row = pd.read_csv(self.data_directory, skiprows=index + 1, nrows=1, header=None)
            tle = row.iloc[0, 6]
            y = row.iloc[0, 0:6].values.astype(np.float32)
            t = float(row.iloc[0, 7])
        
        return tle, y, t

    def __len__(self):
        return self.n_samples

###################################### IMPORTING AND INITIALIZING DATA#########################################
#TRAIN_dir = "mini_train_data.csv"
#TEST_dir = "mini_test_data.csv"
#VALIDATION_dir = "mini_validation_data.csv"

TRAIN_dir = "56794_TRAIN.csv"
TEST_dir = "56794_TEST.csv"
VALIDATION_dir = "56794_VALID.csv"


TRAIN_Dataset = Raj_Dataset(TRAIN_dir,mode ="eager")
TEST_Dataset = Raj_Dataset(TEST_dir,mode ="eager")
VALIDATION_Dataset = Raj_Dataset(VALIDATION_dir,mode ="eager")

TRAIN_dataloader = DataLoader(dataset = TRAIN_Dataset,batch_size = BATCH_SIZES,shuffle = True,num_workers=5,persistent_workers=True)
TEST_dataloader = DataLoader(dataset = TEST_Dataset,batch_size = BATCH_SIZES,shuffle = False,num_workers=5,persistent_workers=True)
VALIDATION_dataloader = DataLoader(dataset = VALIDATION_Dataset,batch_size = BATCH_SIZES,shuffle = False,num_workers=5,persistent_workers=True)

#########################################################################################################################


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#################TRAINING##############################
if __name__ == '__main__':
    RAJ_MODEL = BasicLightning().to(device)
    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        log_every_n_steps=1
    )
    
    trainer.fit(RAJ_MODEL, TRAIN_dataloader, VALIDATION_dataloader)
    trainer.validate(RAJ_MODEL, VALIDATION_dataloader)
    trainer.test(RAJ_MODEL, TEST_dataloader)

    ##logging losses
    df_logs = pd.DataFrame(log_records)
    df_logs.to_csv("56794_results.csv", index=False)
    print("Log saved to training_log_df.csv")

    # >>>> Save the trained model
    model_save_path = "raj_model_56794 again.pt"
    torch.save(RAJ_MODEL.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


# >>>> Function to load the model later

def load_model(checkpoint_path, device=torch.device("cpu")):
    model = BasicLightning()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Example usage after training or in a new script:
# model = load_model("raj_model_checkpoint.pt", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))




