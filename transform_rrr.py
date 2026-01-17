import numpy as np
import torch
from scipy import signal

import math
import cv2

import random



class Transform:
    def __init__(self):
        pass




    def add_noise(self, signal, noise_amount):
        """
        adding noise for Input use only
        """
        signal = signal
        noise = (0.4 ** 0.5) * np.random.normal(1, noise_amount, size=(np.shape(signal)[0],np.shape(signal)[1]))
        #print('noise1',noise.shape)

        #noise = noise[None,:]
        #print('noise2',noise.shape)

        noised_signal = signal + noise

        noised_signal = noised_signal
       # print('noised_signal',noised_signal.shape)
        #noised_signal=noised_signal.transpoe(())
        return noised_signal







    def add_noise_with_SNR(self,signal, noise_amount):
        """
        adding noise
        created using: https://stackoverflow.com/a/53688043/10700812
        """

        # print('singal=',signal.shape)
        # signal = signal
        # print('singal2=',signal.shape)
        target_snr_db = noise_amount  # 20
        x_watts = signal ** 2  # Calculate signal power and convert to dB
        # print('x_watts=',x_watts.shape)
        sig_avg_watts = np.mean(x_watts,axis=1)
        # print('sig_avg_watts=',sig_avg_watts.shape)
        sig_avg_db = 10 * np.log10(sig_avg_watts)  # Calculate noise then convert to watts 
        # print('sig_avg_db=',sig_avg_db.shape)
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts),len(x_watts))  # Generate an sample of white noise
        #print('noise_volts=',noise_volts.shape)
        noise_volts=noise_volts[:,None]
        noised_signal = signal + noise_volts  # noise added signal
        # noised_signal = noised_signal[None,:]
        # print(noised_signal.shape)

        return noised_signal

    # # 
    




    def scaled(self, signal, factor_list):
        
        channels, samples = signal.shape
        Signal=np.zeros((channels,samples))

        # Scale each channel independently
        
        for ch in range(channels):  # Iterate over channels
            factor = round(np.random.uniform(factor_list[0], factor_list[1]), 2)
           
            #signal[ch, :] = 1 / (1 + np.exp(-signal[ch, :])) * factor  # Apply sigmoid and scale
            Signal[ch, :] = signal[ch, :]* 3 # Apply sigmoid and scale

        return Signal


    def DC(self,signal,factor_list):
            """
            negate the signal
            """
            factor = round(np.random.uniform(factor_list[0], factor_list[1]), 2)

            Signal = signal +.5
            #signal=np.squeeze(signal)
            return Signal


    def negate(self,signal):
        """
        negate the signal
        """
        signal = signal * (-1)
        signal=np.squeeze(signal)
        return signal

    

    def hor_flip(self, signal):
        """
        Horizontally flip each channel of a multi-channel signal.
        
        Args:
            signal: numpy array of shape (batch, channels, samples)
        
        Returns:
            Signal with each channel flipped horizontally.
        """
        channels, samples = signal.shape
        result_signal = np.zeros_like(signal)  # Initialize result with the same shape as the input
        
        # Flip each batch and channel independently
       
        for ch in range(channels):  # Iterate over channels
            result_signal[ch, :] = np.flip(signal[ch, :], axis=0)  # Flip the signal along the time axis (axis=0)

        return result_signal
 



 
    def permute(self, signal, pieces):
        """
        Apply a permutation operation to a multi-channel signal.
        
        Args:
            signal: numpy array of shape (batch, channels, samples)
            pieces: Number of segments along the time to permute.
        
        Returns:
            Signal with permuted segments for each batch and channel.
        """
        channels, samples = signal.shape
        result_signal = np.zeros_like(signal)  # Initialize result with the same shape as the input
        
        # Process each batch and channel independently
       
        for ch in range(channels):  # Iterate over channels
            signal_ch = signal[ch, :]

            # Calculate piece size
            piece_length = int(samples // pieces)

            # Generate random permutation of pieces
            sequence = list(range(pieces))
            np.random.shuffle(sequence)

            # Split the signal into pieces
            permuted_signal = np.reshape(signal_ch[:(samples // pieces) * pieces], (pieces, piece_length)).tolist()

            # Handle the tail (remaining part of the signal after dividing into pieces)
            tail = signal_ch[(samples // pieces) * pieces:]

            # Apply permutation
            permuted_signal = np.asarray(permuted_signal)[sequence]
            permuted_signal = np.concatenate(permuted_signal, axis=0)  # Concatenate the permuted pieces
            permuted_signal = np.concatenate((permuted_signal, tail), axis=0)  # Add the tail to the result

            # Store the permuted signal in the result
            result_signal[ch, :] = permuted_signal


        return result_signal
    


    def cutout_resize(self, signal, pieces):
        """
        Apply cutout and resize operations to a multi-channel signal.
        
        Args:
            signal: numpy array of shape (batch, channels, samples)
            pieces: Number of segments along the time to apply the cutout operation.
        
        Returns:
            Signal with one randomly cutout segment for each batch and channel, resized to 3072 samples.
        """
        channels, samples = signal.shape
        result_signal = np.zeros_like(signal)  # Initialize result with the same shape as the input
        
        
       
        for ch in range(channels):  # Iterate over channels
            signal_ch = signal[ch, :]

            # Calculate piece size
            piece_length = int(samples // pieces)

            # Generate random cutout segment
            cutout = random.randint(0, pieces - 1)  # Randomly choose a segment to cutout
            sequence = [i for i in range(pieces) if i != cutout]  # Create a sequence excluding the cutout segment

            # Split signal into pieces and apply cutout
            cutout_signal = np.reshape(signal_ch[:(samples // pieces) * pieces], (pieces, piece_length)).tolist()
            tail = signal_ch[(samples // pieces) * pieces:]

            cutout_signal = np.asarray(cutout_signal)[sequence]  # Remove the cutout piece
            cutout_signal = np.hstack(cutout_signal)  # Concatenate the remaining pieces
            cutout_signal = np.concatenate((cutout_signal, tail), axis=0)  # Add the tail

            # Resize the signal to the target length (3072 samples)
            cutout_signal = cv2.resize(cutout_signal.reshape(-1, 1), (1, samples), interpolation=cv2.INTER_LINEAR).flatten()
            result_signal[ch, :] = cutout_signal  # Store the result in the output array
    
        

        return result_signal
    

    def cutout_zero(self, signal, pieces):
        """
        Apply a cutout operation to a multi-channel signal.
        
        Args:
            signal: numpy array of shape (batch, channels, samples)
            pieces: Number of segments along time to perform the cutout.
        
        Returns:
            Signal with one randomly cutout segment for each batch and channel.
        """
        channels, samples = signal.shape
        result_signal = np.zeros_like(signal)  # Initialize result with the same shape as the input
        
        for ch in range(channels):  # Iterate over channels
            signal_ch = signal[ch, :]

            # Calculate piece size
            piece_length = int(samples // pieces)
            ones = np.ones(samples)

            # Generate segments
            cutout = random.randint(1, pieces)
            cutout_signal = np.reshape(signal_ch[:(samples // pieces) * pieces], (pieces, piece_length)).tolist()
            ones_pieces = np.reshape(ones[:(samples // pieces) * pieces], (pieces, piece_length)).tolist()
            tail = signal_ch[(samples // pieces) * pieces:]

            cutout_signal = np.asarray(cutout_signal)
            ones_pieces = np.asarray(ones_pieces)
            
            # Apply cutout (set one segment to zero)
            for i in range(pieces):
                if i == cutout - 1:  # -1 because of zero indexing
                    ones_pieces[i] *= 0

            cutout_signal = cutout_signal * ones_pieces
            cutout_signal = np.hstack(cutout_signal)
            cutout_signal = np.concatenate((cutout_signal, tail), axis=0)

            result_signal[ch, :] = cutout_signal  # Store result in the output array

        return result_signal
    






    

    
    def crop_resize(self, signal, size):
        """
        Crop and resize multi-channel signals.

        Args:
            signal: numpy array of shape (batch, channels, samples).
            size: Proportion of the signal to retain after cropping (e.g., 0.5 for 50%).
        
        Returns:
            Resized signal with the same shape as input.
        """
        channels, samples = signal.shape
        resized_signal = np.zeros((channels, samples))  # Output shape has 3072 samples
        for ch in range(channels):  # Iterate over channels
            # Crop the signal
            crop_size = int(samples * size)
            start = random.randint(0, samples - crop_size)
            crop_signal = signal[ch, start:start + crop_size]

            # Resize the cropped signal to the target length
            resized_signal[ch, :] = cv2.resize(crop_signal.reshape(-1, 1), (1, samples), interpolation=cv2.INTER_LINEAR).flatten()

        return resized_signal






    def move_avg(self,a,n, mode="same"):
        # a = a.T
        channels, samples = a.shape
        result = np.zeros_like(a)
        for ch in range(channels):  # Iterate over channels
            result[ch, :] = np.convolve(a[ch, :], np.ones((n,)) / n, mode=mode)
                
        #result=np.squeeze(result)
        return result
    



    def bandpass_filter(self, x, order, cutoff, fs=100):
        result = np.zeros((x.shape[0], x.shape[1]))
        w1 = 2 * cutoff[0] / int(fs)
        w2 = 2 * cutoff[1] / fs
        b, a = signal.butter(order, [w1, w2], btype='bandpass')  # 配置滤波器 8 表示滤波器的阶数
        for chnl in range(x.shape[0]):
             
             result[chnl,:] = signal.filtfilt(b, a, x[chnl,:])
        # print(result.shape)
        #result=np.squeeze(result)
        return result
    


    def lowpass_filter(self, x, order, cutoff, fs=100):
        result = np.zeros((x.shape[0], x.shape[1]))
        w1 = 2 * cutoff[0] / int(fs)
        # w2 = 2 * cutoff[1] / fs
        b, a = signal.butter(order, w1, btype='lowpass')  # 配置滤波器 8 表示滤波器的阶数
        for chnl in range(x.shape[0]):
             
            result[chnl,:] = signal.filtfilt(b, a, x[chnl,:])
        # print(result.shape)
        #result=np.squeeze(result)
        return result

    def highpass_filter(self, x, order, cutoff, fs=100):

        result = np.zeros((x.shape[0], x.shape[1]))
        w1 = 2 * cutoff[0] / int(fs)
        # w2 = 2 * cutoff[1] / fs
        b, a = signal.butter(order, w1, btype='highpass')  # 配置滤波器 8 表示滤波器的阶数

        for chnl in range(x.shape[0]):
             
             result[chnl,:] = signal.filtfilt(b, a, x[chnl,:])
        result=np.squeeze(result)
        return result


    
    def time_warp(self, signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
        """
        Apply time-warping to a multi-channel signal.
        
        Args:
            signal: numpy array (batch x channels x window)
            sampling_freq: Sampling frequency of the signal.
            pieces: Number of segments along time.
            stretch_factor: Factor by which to stretch segments.
            squeeze_factor: Factor by which to squeeze segments.
        
        Returns:
            Time-warped signal of the same shape as the input.
        """
        # Initialize the output signal with the same shape as input
        channels, total_samples = signal.shape
        time_warped_signal = np.zeros_like(signal)

        total_time = total_samples / sampling_freq
        segment_time = total_time / pieces
        sequence = list(range(0, pieces))
        stretch = np.random.choice(sequence, math.ceil(len(sequence) / 2), replace=False)
        squeeze = list(set(sequence).difference(set(stretch)))

        
        for ch in range(channels):  # Iterate over channels
            # Process each channel independently
            channel_signal = signal[ch, :].reshape(-1, 1)  # Reshape to (samples, 1)
            initialize = True
            for i in sequence:
                # Extract segment
                start_idx = int(i * np.floor(segment_time * sampling_freq))
                end_idx = int((i + 1) * np.floor(segment_time * sampling_freq))
                orig_signal = channel_signal[start_idx:end_idx]

                # Stretch or squeeze the segment
                if i in stretch:
                    output_shape = int(np.ceil(orig_signal.shape[0] * stretch_factor))
                elif i in squeeze:
                    output_shape = int(np.ceil(orig_signal.shape[0] * squeeze_factor))

                # Resize the segment
                new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)

                # Concatenate the warped segments
                if initialize:
                    warped_channel = new_signal
                    initialize = False
                else:
                    warped_channel = np.vstack((warped_channel, new_signal))

            # Resize the warped signal to the original length
            warped_channel = cv2.resize(warped_channel, (1, total_samples), interpolation=cv2.INTER_LINEAR)
            time_warped_signal[ch, :] = warped_channel.flatten()
        #time_warped_signal=np.squeeze(time_warped_signal)
        return time_warped_signal
    




#_____________________________________________________________________________________________    

if __name__ == '__main__':
    from transform_rrr import Transform
    import matplotlib.pyplot as plt


    chnum=22
    fs=100

    Trans = Transform()
    input = np.zeros((chnum,1000))
    print('inpuptshap1',input.shape)

    # input = Trans.add_noise(input,10)
    # print('inpuptshap2',input.shape)

    
    f=1
    T=3
    fs=100
    t=np.arange(0,T,1/fs)
    x0=np.sin(2*np.pi*f*t)
    x1=np.sin(2*np.pi*1*f*t)

    input=np.stack((x0,x1),axis=0)
    print('inpuptshap2',input.shape)
    

##_________________________________________    
    
    # order = random.randint(3,10)
    # cutoff = random.uniform(5, 20)

    # output= Trans.lowpass_filter(input, order, [2,15], fs)

##_______________________________________________
    # pieces = random.randint(5,20)
    # stretch = random.uniform(1.5,4)
    # squeeze = random.uniform(0.25,0.67)

    # output=Trans.time_warp(input,fs,pieces,stretch,squeeze)
##______________________________________________________-    

    # n = random.randint(3, 10)
    # output= Trans.move_avg(input,n, mode="same")

##__________________________________________________

    # size=0.75
    # output=Trans.crop_resize(input, size)
##__________________________________________________________


    # num_pieces = 10
    # output = Trans.cutout_zero(input, num_pieces)
##_______________________________________________________
    
    # pieces = 10
    # output=Trans.cutout_resize( input, pieces)

    
##______________________________________________________-

    # pieces = 10
    # output=Trans.permute(input,pieces)
##_____________________________________________________


   # output=Trans.hor_flip(input)
##________________________________________________________

    
    #output=Trans.negate(input)
##___________________________________________________________-

    factor_list=[2, 4]
    output=Trans.scaled(input,factor_list)

##______________________________________________________
# 
# 
    factor_list=[1, 4]
    output=Trans.DC(input,factor_list)    


##___________________________________________


   # output=Trans.add_noise_with_SNR(input,10)

##_____________________________________________
    print('outputshap',output.shape)

##_____________________________________________
    plt.subplot(211)
    plt.plot(input[0,:])
    plt.plot(output[0,:])
    plt.legend('IO')

    plt.subplot(212)
    plt.plot(input[1,:])
    plt.plot(output[1,:])
    plt.show()
    #plt.savefig('filter.png')
    print('outputsha',output.shape)
