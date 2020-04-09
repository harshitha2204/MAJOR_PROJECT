import os
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import wave



def remove_silence(filename, out_dir, smoothing=1.0, weight=0.3, plot=False):
    partic_id = 'P' + filename.split('/')[-1].split('_')[0]  # PXXX
    if is_segmentable(partic_id):
        # create participant directory for segmented wav files
        participant_dir = os.path.join(out_dir, partic_id)
        #if not os.path.exists(participant_dir):
            #os.makedirs(participant_dir)
        participant_dir=participant_dir+'_AUDIO'
        print(participant_dir) 
        os.chdir(participant_dir)

        [Fs, x] = aIO.readAudioFile(filename)
        segments = aS.silenceRemoval(x, Fs, 0.020, 0.020,
                                     smoothWindow=smoothing,
                                     Weight=weight,
                                     plot=plot)

        for s in segments:
            seg_name = "{:s}_{:.2f}-{:.2f}.wav".format(partic_id, s[0], s[1])
            wavfile.write(seg_name, Fs, x[int(Fs * s[0]):int(Fs * s[1])])

        # concatenate segmented wave files within participant directory
        concatenate_segments(participant_dir, partic_id)


def is_segmentable(partic_id):
   
    troubled = set(['P300', 'P305', 'P306', 'P308', 'P315', 'P316', 'P343',
                    'P354',])
    return partic_id not in troubled


def concatenate_segments(participant_dir, partic_id, remove_segment=True):
    
    infiles = os.listdir(participant_dir)  # list of wav files in directory
    outfile = '{}_no_silence.wav'.format(partic_id)

    data = []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
        if remove_segment:
            os.remove(infile)

    output = wave.open(outfile, 'wb')
    
    output.setparams(data[0][0])

    for idx in range(len(data)):
        output.writeframes(data[idx][1])
    output.close()


if __name__ == '__main__':
    # directory containing raw wav files
    dir_name = r"C:\Users\Harini\Documents\PROJECT\src\raw\audio"


    # directory where a participant folder will be created containing their
    # segmented wav file
    out_dir = r"C:\Users\Harini\Documents\PROJECT\src\interim"
    


    # iterate through wav files in dir_name and create a segmented wav_file
    for file in os.listdir(dir_name):
        if file.endswith('.wav'):
            filename = os.path.join(dir_name, file)
            remove_silence(filename, out_dir)
