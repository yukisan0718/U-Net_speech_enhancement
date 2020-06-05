#!/usr/bin/env python
# coding: utf-8

### Import libraries ###
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
import os
import sys
import time
import glob
import gc
import h5py
import math
import random
from tensorflow import config
from tensorflow.keras import backend
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from datetime import datetime
from pystoi import stoi as STOI
from pesq import pesq as PESQ

### Function for pre-processing ###
def pre_processing(data, Fs, down_sample):
    
    #Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    
    #Downsample if necessary
    if down_sample is not None:
        wavdata = sg.resample_poly(wavdata, down_sample, Fs)
        Fs = down_sample
    
    return wavdata, Fs

### Function for calculating STFT-Spectrogram ###
def get_STFT(folder_path, down_sample, frame_length, frame_shift, num_frames):
    
    #Initialize list
    x = []
    
    #Get .wav files as an object
    files = sorted(glob.glob(folder_path + "/*.wav"))
    print("Folder:" + folder_path)
    
    #For a progress bar
    nfiles = len(files)
    unit = math.floor(nfiles/20)
    bar = "#" + " " * math.floor(nfiles/unit)
    
    #Repeat every file-name
    for i, file in enumerate(files):
        
        #Display a progress bar
        print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
        if i % unit == 0:
            bar = "#" * math.ceil(i/unit) + " " * math.floor((nfiles-i)/unit)
            print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
        
        #Read .wav file and get pre-process
        wavdata, Fs = sf.read(file)
        wavdata, Fs = pre_processing(wavdata, Fs, down_sample)
        
        #Calculate the index of window size and overlap
        FL = round(frame_length * Fs)
        FS = round(frame_shift * Fs)
        OL = FL - FS
        
        #Execute STFT
        _, _, dft = sg.stft(wavdata, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
        dft = dft[:-1].T #Remove the last point and get transpose
        spec = np.log10(np.abs(dft))
        
        #Crop the temporal frames into input size
        num_seg = math.floor(spec.shape[0] / num_frames)
        for j in range(num_seg):
            #Add results to list sequentially
            x.append(spec[int(j*num_frames) : int((j+1)*num_frames), :])
    
    #Finish the progress bar
    bar = "#" * math.ceil(nfiles/unit)
    print("\rProgress:[{0}] {1}/{2} Completed!   ".format(bar, i+1, nfiles), end="")
    print()
    
    #Convert into numpy array
    x = np.array(x)
    
    #Return the result
    return x, Fs

### Function for reading an audio and getting the STFT ###
def read_evaldata(file_path, down_sample, frame_length, frame_shift, num_frames):
    
    #Inicialize list
    x = []
    ang_x = []
    
    #Read .wav file and get pre-process
    wavdata, Fs = sf.read(file_path)
    wavdata, Fs = pre_processing(wavdata, Fs, down_sample)
    
    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Execute STFT
    _, _, dft = sg.stft(wavdata, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    dft = dft[:-1].T #Remove the last point and get transpose
    ang = np.angle(dft) #Preserve the phase
    spec = np.log10(np.abs(dft))
    
    #Crop the temporal frames into input size
    num_seg = math.floor(spec.shape[0] / num_frames)
    for j in range(num_seg):
        #Add results to list sequentially
        x.append(spec[int(j*num_frames) : int((j+1)*num_frames), :])
        ang_x.append(ang[int(j*num_frames) : int((j+1)*num_frames), :])
    
    #Convert into numpy array
    x = np.array(x)
    ang_x = np.array(ang_x)
    
    return wavdata, Fs, x, ang_x

### Function for reconstructing a waveform ###
def reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift):
    
    #Construct the spectrogram by concatenating all segments
    Y = np.reshape(eval_y, (-1, eval_y.shape[-1]))
    ang = np.reshape(ang_x, (-1, ang_x.shape[-1]))
    
    #The Y and arg can be transpose for Tensorflow format
    Y, ang = Y.T, ang.T
    
    #Restore the magnitude of STFT
    Y = np.power(10, Y)
    
    #Restrive the phase from original wave
    Y = Y * np.exp(1j*ang)
    
    #Add the last frequency bin along with frequency axis
    Y = np.append(Y, Y[-1, :][np.newaxis,:], axis=0)
    
    #Get the inverse STFT
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    _, rec_wav = sg.istft(Y, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    
    return rec_wav, Fs

### Function to change the learning rate for each epoch ###
def step_decay(x):
    y = learn_rate * 10**(-lr_decay*x)
    return y

### Function for computing LSD loss (LSD: log-spectral distance) ###
def LSD_loss(y_true, y_pred):
    LSD = backend.mean((y_true - y_pred)**2, axis=2)
    LSD = backend.mean(backend.sqrt(LSD), axis=1)
    return LSD

### Function for executing CNN learning ###
def UNet_learning(train_x, train_y, test_x, test_y, LR, BS, EP):
    
    #Memory saving
    devices = config.experimental.list_physical_devices('GPU')
    if len(devices) > 0:
        for k in range(len(devices)):
            config.experimental.set_memory_growth(devices[k], True)
            print('memory growth:', config.experimental.get_memory_growth(devices[k]))
    else:
        print("Not enough GPU hardware devices available")
    
    #Path for saving CNN model
    p1 = "./models/model.json"
    p2 = "./models/weights.h5"
    
    #If the path exist
    if True:
        
        #Change the learning phase into training mode
        backend.set_learning_phase(1)
        
        #Add a color dimension to input (for Tensorflow format)
        train_x = train_x[:, :, :, np.newaxis]
        train_y = train_y[:, :, :, np.newaxis]
        test_x = test_x[:, :, :, np.newaxis]
        test_y = test_y[:, :, :, np.newaxis]
        
        #Get the number of row and column in input
        row = train_x.shape[1]
        column = train_x.shape[2]
        print("input_data_shape: " + str(train_x.shape) )
        
        #Define the input size(row, column, dimension)
        image_layer = Input(shape=(row, column, 1))
        
        #Construct the U-Net model with Functional API by Keras
        enc1 = Conv2D(32, kernel_size=(5, 7), strides=(1, 2), padding='same')(image_layer)
        enc1 = BatchNormalization()(enc1)
        enc1 = LeakyReLU(alpha=0.2)(enc1)
        
        enc2 = Conv2D(64, kernel_size=(5, 7), strides=(1, 2), padding='same')(enc1)
        enc2 = BatchNormalization()(enc2)
        enc2 = LeakyReLU(alpha=0.2)(enc2)
        
        enc3 = Conv2D(128, kernel_size=(5, 7), strides=(1, 2), padding='same')(enc2)
        enc3 = BatchNormalization()(enc3)
        enc3 = LeakyReLU(alpha=0.2)(enc3)
        
        enc4 = Conv2D(256, kernel_size=(5, 5), strides=(1, 2), padding='same')(enc3)
        enc4 = BatchNormalization()(enc4)
        enc4 = LeakyReLU(alpha=0.2)(enc4)
        
        enc5 = Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(enc4)
        enc5 = BatchNormalization()(enc5)
        enc5 = LeakyReLU(alpha=0.2)(enc5)
        
        enc6 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc5)
        enc6 = BatchNormalization()(enc6)
        enc6 = LeakyReLU(alpha=0.2)(enc6)
        
        enc7 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc6)
        enc7 = BatchNormalization()(enc7)
        enc7 = LeakyReLU(alpha=0.2)(enc7)
        
        enc8 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc7)
        enc8 = BatchNormalization()(enc8)
        enc8 = LeakyReLU(alpha=0.2)(enc8)
        
        dec1 = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc8)
        dec1 = BatchNormalization()(dec1)
        dec1 = ReLU()(dec1)
        dec1 = Dropout(0.5)(dec1)
        
        dec2 = concatenate([dec1, enc7], axis=-1)
        dec2 = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(dec2)
        dec2 = BatchNormalization()(dec2)
        dec2 = ReLU()(dec2)
        dec2 = Dropout(0.5)(dec2)
        
        dec3 = concatenate([dec2, enc6], axis=-1)
        dec3 = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(dec3)
        dec3 = BatchNormalization()(dec3)
        dec3 = ReLU()(dec3)
        dec3 = Dropout(0.5)(dec3)
        
        dec4 = concatenate([dec3, enc5], axis=-1)
        dec4 = Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(dec4)
        dec4 = BatchNormalization()(dec4)
        dec4 = ReLU()(dec4)
        
        dec5 = concatenate([dec4, enc4], axis=-1)
        dec5 = Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 2), padding='same')(dec5)
        dec5 = BatchNormalization()(dec5)
        dec5 = ReLU()(dec5)
        
        dec6 = concatenate([dec5, enc3], axis=-1)
        dec6 = Conv2DTranspose(64, kernel_size=(5, 7), strides=(1, 2), padding='same')(dec6)
        dec6 = BatchNormalization()(dec6)
        dec6 = ReLU()(dec6)
        
        dec7 = concatenate([dec6, enc2], axis=-1)
        dec7 = Conv2DTranspose(32, kernel_size=(5, 7), strides=(1, 2), padding='same')(dec7)
        dec7 = BatchNormalization()(dec7)
        dec7 = ReLU()(dec7)
        
        dec8 = concatenate([dec7, enc1], axis=-1)
        dec8 = Conv2DTranspose(1, kernel_size=(5, 7), strides=(1, 2), padding='same', activation='sigmoid')(dec8)
        
        #Construct the model and display summary
        cnn_model = Model(image_layer, dec8)
        print(cnn_model.summary())
        
        #Define the optimizer (SGD with momentum or Adam)
        opt = Adam(lr=LR, beta_1=0.5, beta_2=0.9)
        
        #Compile the model (calling LSD_loss function)
        cnn_model.compile(loss=LSD_loss, optimizer=opt)
        
        #Start learning
        lr_decay = LearningRateScheduler(step_decay)
        hist = cnn_model.fit(train_x, train_y, batch_size=BS, epochs=EP, validation_data=(test_x, test_y), callbacks=[lr_decay], verbose=1)
        
        #Save the weights of learned model
        model_json = cnn_model.to_json()
        with open(p1, 'w') as f:
            f.write(model_json)
        cnn_model.save_weights(p2)
        
        #Save the learning history as text file
        log_path = "./log/loss_function.txt"
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        with open(log_path, "a") as fp:
            fp.write("epoch\tloss\tval_loss\n")
            for i in range(len(loss)):
                fp.write("%d\t%f\t%f" % (i, loss[i], val_loss[i]))
                fp.write("\n")
        
        #Display the learning history
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(18, 5))
        plt.plot(hist.history['loss'], label="loss for training")
        plt.plot(hist.history['val_loss'], label="loss for validation")
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        #Save the graph
        plt.savefig("./log/loss_function.png", format="png", dpi=300)
        plt.show()
        
        #Restart the session to relieve the GPU memory (to prevent Resource Exhausted Error)
        backend.clear_session()
        #backend.get_session() #Less than tensorflow ver.1.14
        del cnn_model
        gc.collect()
    
    return

### Function for executing CNN learning ###
def UNet_evaluation(eval_x):
    
    #Path for saving CNN model
    p1 = "./models/model.json"
    p2 = "./models/weights.h5"
    
    #If the path exist
    if os.path.isfile(p1) and os.path.isfile(p2):
        
        #Change the learning phase into test mode
        backend.set_learning_phase(0)
        
        #Read the pre-learned model and its weight
        with open(p1, "r") as f:
            cnn_model = model_from_json(f.read())
        cnn_model.load_weights(p2)
        
        #Add a color dimension to input (for Tensorflow format)
        x = eval_x[:, :, :, np.newaxis]
        
        #Get the separated source for evaluation data
        y = cnn_model.predict(x)
        #y = cnn.model.predict_on_batch(x)
        
        #Remove a color dimension from input (for Tensorflow format)
        eval_y = y[:,:,:,0]
    
    #Restart the session to relieve the GPU memory (to prevent Resource Exhausted Error)
    backend.clear_session()
    #backend.get_session() #Less than tensorflow ver.1.14
    del cnn_model
    gc.collect()
    
    return eval_y

### Main ###
if __name__ == "__main__":
    
    #Set up
    down_sample = 16000    #Downsampling rate (Hz) [Default]16000
    frame_length = 0.032   #STFT window width (second) [Default]0.032
    frame_shift = 0.016    #STFT window shift (second) [Default]0.016
    num_frames = 16        #The number of frames for an input [Default]16
    learn_rate = 1e-4      #Lerning rate for CNN training [Default]1e-4
    lr_decay = 0           #Lerning rate is according to "learn_rate*10**(-lr_decay*n_epoch)" [Default]0
    batch_size = 64        #Size of batch for CNN training [Default]64
    epoch = 30             #The number of iteration for CNN training [Default]30
    mode = "train"         #Select either "train" or "eval" [Default]"train"
    stft = True            #True: compute STFT from the beginning, False: read numpy files [Default]True
    num_sample = 800        #The number of samples for evaluation [Default]800
    
    #For training
    if mode == "train":
        
        #In case of computing the STFT at the beginning
        if stft == True:
            #Compute STFT for the mixed source
            fpath = "./audio_data/training/NOISY"
            train_x, Fs = get_STFT(fpath, down_sample, frame_length, frame_shift, num_frames)
            
            #Compute STFT for the separated source
            fpath = "./audio_data/training/CLEAN"
            train_y, Fs = get_STFT(fpath, down_sample, frame_length, frame_shift, num_frames)
            
            #Save the training data
            fpath = "./numpy_files"
            np.save(fpath + '/X_train', train_x)
            np.save(fpath + '/Y_train', train_y)
        
        #In case of reading the STFT spectrogram from local file
        else:
            #Read the training data
            fpath = "./numpy_files"
            train_x = np.load(fpath + '/X_train.npy')
            train_y = np.load(fpath + '/Y_train.npy')
        
        #Remove segments including -inf entries in train_x
        idx = np.unique(np.where(train_x == -np.inf)[0])
        idx = list(set(range(train_x.shape[0])) - set(idx)) #Remove indices of -inf
        train_x = train_x[idx, :, :]
        train_y = train_y[idx, :, :]
        
        #Remove segments including -inf entries in train_y
        idx = np.unique(np.where(train_y == -np.inf)[0])
        idx = list(set(range(train_y.shape[0])) - set(idx)) #Remove indices of -inf
        train_x = train_x[idx, :, :]
        train_y = train_y[idx, :, :]
        
        #Split the input data into a training set and a small test set
        test_x = train_x[:3000, :, :]
        test_y = train_y[:3000, :, :]
        train_x = train_x[3000:, :, :]
        train_y = train_y[3000:, :, :]
        
        #Normalization (not standardization)
        max_x = np.amax(train_x, axis=None)
        min_x = np.amin(train_x, axis=None)
        print(max_x)
        print(min_x)
        train_x = (train_x - min_x) / (max_x - min_x)
        train_y = (train_y - min_x) / (max_x - min_x)
        test_x = (test_x - min_x) / (max_x - min_x)
        test_y = (test_y - min_x) / (max_x - min_x)
        
        #Call my function for executing CNN learning
        UNet_learning(train_x, train_y, test_x, test_y, learn_rate, batch_size, epoch)
        print("Done")
    
    #For evaluation
    elif mode == "eval":
        
        #Compute STFT for the mixed source
        fpath = "./audio_data/evaluation/NOISY"
        print("Folder:" + fpath)
        
        #Get .wav files as an object
        files = sorted(glob.glob(fpath + "/*.wav"))
        samples = random.sample(list(range(len(files))), k=num_sample) #Extract samples randomly
        
        #Define valuables for metrics
        nfiles = len(samples)
        PESQ_mix, STOI_mix, ESTOI_mix = np.zeros(nfiles), np.zeros(nfiles), np.zeros(nfiles)
        PESQ_sep, STOI_sep, ESTOI_sep = np.zeros(nfiles), np.zeros(nfiles), np.zeros(nfiles)
        
        #For a progress bar
        if nfiles >= 20:
            unit = math.floor(nfiles/20)
        else:
            unit = math.floor(nfiles/1)
        bar = "#" + " " * math.floor(nfiles/unit)
        
        #Repeat for each file
        for i, sample in enumerate(samples):
            
            #Display a progress bar
            print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
            if i % unit == 0:
                bar = "#" * math.ceil(i/unit) + " " * math.floor((nfiles-i)/unit)
                print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
            
            #Call my fucntion for reading audio
            mix_wav, Fs, eval_x, ang_x = read_evaldata(files[sample], down_sample, frame_length, frame_shift, num_frames)
            
            #Normalization
            max_x = -0.2012536819610933 #From training step
            min_x = -9.188868273733446 #From training step
            eval_x = (eval_x - min_x) / (max_x - min_x)
            
            #Call my function for separating audio by the pre-learned U-Net model
            eval_y = UNet_evaluation(eval_x)
            
            #Restore the scale before normalization
            eval_y = eval_y * (max_x - min_x) + min_x
            
            #Call my function for reconstructing the waveform
            sep_wav, Fs = reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift)
            
            #Read the ground truth
            CLEAN_path = files[sample].replace(fpath.split("/")[-1], 'CLEAN')
            clean_wav, Fs = sf.read(CLEAN_path)
            clean_wav, Fs = pre_processing(clean_wav, Fs, down_sample)
            
            #Adjust the length of audio
            diff = int(mix_wav.shape[0]) - int(sep_wav.shape[0])
            if diff > 0:
                mix_wav = mix_wav[:-diff]
                clean_wav = clean_wav[:-diff]
            else:
                sep_wav = sep_wav[:diff]
            
            #Compute the PESQ and STOI scores
            PESQ_mix[i] = PESQ(Fs, clean_wav, mix_wav, 'wb')
            STOI_mix[i] = STOI(clean_wav, mix_wav, Fs, extended=False)
            ESTOI_mix[i] = STOI(clean_wav, mix_wav, Fs, extended=True)
            PESQ_sep[i] = PESQ(Fs, clean_wav, sep_wav, 'wb')
            STOI_sep[i] = STOI(clean_wav, sep_wav, Fs, extended=False)
            ESTOI_sep[i] = STOI(clean_wav, sep_wav, Fs, extended=True)
            #print("PESQ(mix): {:.4f}, STOI(mix): {:.4f}, ESTOI(mix): {:.4f}".format(PESQ_mix[i], STOI_mix[i], ESTOI_mix[i]))
            #print("PESQ(sep): {:.4f}, STOI(sep): {:.4f}, ESTOI(sep): {:.4f}".format(PESQ_sep[i], STOI_sep[i], ESTOI_sep[i]))
        
        #Finish the progress bar
        bar = "#" * math.ceil(nfiles/unit)
        print("\rProgress:[{0}] {1}/{2} Completed!   ".format(bar, i+1, nfiles), end="")
        print()
        
        #Compute the average scores
        avePESQ_mix, aveSTOI_mix, aveESTOI_mix = np.mean(PESQ_mix), np.mean(STOI_mix), np.mean(ESTOI_mix)
        avePESQ_sep, aveSTOI_sep, aveESTOI_sep = np.mean(PESQ_sep), np.mean(STOI_sep), np.mean(ESTOI_sep)
        print("PESQ(original): {:.4f}, STOI(original): {:.4f}, ESTOI(original): {:.4f}".format(avePESQ_mix, aveSTOI_mix, aveESTOI_mix))
        print("PESQ(separated): {:.4f}, STOI(separated): {:.4f}, ESTOI(separated): {:.4f}".format(avePESQ_sep, aveSTOI_sep, aveESTOI_sep))
        
        #Save as a log file
        with open("./log/evaluation.txt", "w") as f:
            f.write("PESQ(original): {:.4f}, STOI(original): {:.4f}, ESTOI(original): {:.4f}\n".format(avePESQ_mix, aveSTOI_mix, aveESTOI_mix))
            f.write("PESQ(separated): {:.4f}, STOI(separated): {:.4f}, ESTOI(separated): {:.4f}\n".format(avePESQ_sep, aveSTOI_sep, aveESTOI_sep))