import numpy as np
from numpy.lib import stride_tricks
import os
from PIL import Image
import scipy.io.wavfile as wav




def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0]*hopSize,
                                      samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i+1]], axis=1)

    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs


def stft_matrix(audiopath, binsize=2**10, png_name='tmp.png',save_png=False, offset=0):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    ims = np.flipud(ims)  # weird - not sure why it needs flipping

    if save_png:
        create_png(ims, png_name)

    return ims


def create_png(im_matrix, png_name):
    image = Image.fromarray(im_matrix)
    image = image.convert('L')  # convert to grayscale
    image.save(png_name)


if __name__ == '__main__':
    # directory containing participant folders with segmented wav files
    dir_name = '/Home/Documents/depression-detection/src/interim'

    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('.wav'):
                wav_file = os.path.join(subdir, file)
                png_name = subdir + '/' + file[:-4] + '.png'
                stft_matrix(wav_file, png_name=png_name, save_png=True)
