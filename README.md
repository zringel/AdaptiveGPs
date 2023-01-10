# AdaptiveGPs
Contains all the code relevant for the results reported in 
"Separation of Scales and a Thermodynamics Description of Feature Learning in Some CNNs" 
with regards to DNNs in the mean-field scaling.

1. runner.sh is a bash script designed to run the actual DNN training experiments and gather statistics  
2. Teacher_student_GDNoise_fixed_sched.py is a python script which generates the artificial dataset and performs and GD+noise (unadjusted Lagenvin samples) with a fixed learning rate scheduler
3. noisy_sgd.py is a python library used by item 2. for adding random noise on the gradients. 
4. Solver_3Layer_ErfFCN_Jax-Final_Version.ipynb is a Jupyter notebook which generates the dataset used in the FCN experiments, calculates the emergent scale, and runs a Jax-based solver for our Equations of State.  
