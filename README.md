
# OC-NN-Pcamv
We apply the One-Class Neural Networks (https://arxiv.org/pdf/1802.06360.pdf) for Novelty detection in the Pcam dataset (https://github.com/basveeling/pcam).

`autoencoder_train.py` is the script used to train the autoencoder. The pre-trained autoencoder, with fixed weights, is used to provide a compressed representation that is then used during training and inference of the one-class neural network (OC-NN).  The autoencoder is trained on a subset of the normal training set found in the folder `/pcamv1`.

`OCNN_synthetic.py` is the script used for training of the OC-NN on a synthetic dataset. 

`OCNN_with_Encoder.py` loads the pre-trained encoder from the `/model_run_1621_lr=1e-4_dropout=0` folder and then trains an OC-NN using a subset of the normal training set found in the `/pcamv1` folder. Validation is performed on a subsample of the validation set found in the same containing folder.
