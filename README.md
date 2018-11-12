###
# Authors: Jacob Salam i6184256, Omendra Manhar - i6131589 
###

- The submission already contains the trained weights for different number of epochs, these trained weights can be found in the folder "Old Training Checkpoints". In order to use them, you can simply copy and paste the the contents to "training_checkpoints" folder.

- In order to generate text, you can simply execute "Generate_Text.py" file, and it will use the weights that are in "training_checkpoints" folder to generate texts. The temperature can be varied at line no. 135. 
	- You can vary the start_string at line no. 123 to conduct experiments.
	- If you run it as it is, it will use the weights of RNN after 25 epochs.

- If you want to Train the RNN with new parameters, the "Train.py" file can be executed and "EPOCHS" parameter can be changed at line no. 114

- The dataset used are in txt files, and they are converted to list objects and saved in all_d.list as pickle object.

