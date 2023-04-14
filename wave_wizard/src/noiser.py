import glob
import numpy as np
import soundfile as sf
import os
import argparse
import yaml
import configparser as CP

# Function to read audio
def audioread(path, norm = True, start=0, stop=None):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, sr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(x.shape) == 1:  # mono
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr
    else:  # multi-channel
        x = x.T
        x = x.sum(axis=0)/x.shape[0]
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr
    
# Funtion to write audio    
def audiowrite(data, fs, destpath, norm=False, eps=1e-8):
    if norm:
        rms = (data ** 2).mean() ** 0.5
        scalar = 10 ** (-25 / 10) / (rms+eps)
        data = data * scalar
        if max(abs(data))>=1:
            data = data/max(abs(data), eps)
    
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    sf.write(destpath, data, fs)
    return

# Function to mix clean speech and noise at various SNR levels
def snr_mixer(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean
    rmsclean = (clean**2).mean()**0.5

    rmsnoise = (noise**2).mean()**0.5
    scalarnoise = 10 ** (-25 / 20) /rmsnoise
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5
    
    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return clean, noisenewlevel, noisyspeech

def main(config):
    snr_lower = float(config["snr_lower"])
    snr_upper = float(config["snr_upper"])
    total_snrlevels = float(config["total_snrlevels"])
    
    clean_dir = os.path.join(os.path.dirname(__file__), 'clean_train')
    if config["speech_dir"]:
        clean_dir = config["speech_dir"]
    if not os.path.exists(clean_dir):
        assert False, ("Clean speech data is required")
    
    noise_dir = os.path.join(os.path.dirname(__file__), 'noise_train')
    if config["noise_dir"]:
        noise_dir = config["noise_dir"]
    if not os.path.exists(noise_dir):
        assert False, ("Noise data is required")
        
    fs = float(config["sampling_rate"])
    audioformat = config["audioformat"]
    total_hours = float(config["total_hours"])
    audio_length = float(config["audio_length"])
    silence_length = float(config["silence_length"])
    noisyspeech_dir = os.path.join(os.path.dirname(__file__), 'NoisySpeech_training')
    if not os.path.exists(noisyspeech_dir):
        os.makedirs(noisyspeech_dir)
    clean_proc_dir = os.path.join(os.path.dirname(__file__), 'CleanSpeech_training')
    if not os.path.exists(clean_proc_dir):
        os.makedirs(clean_proc_dir)
    noise_proc_dir = os.path.join(os.path.dirname(__file__), 'Noise_training')
    if not os.path.exists(noise_proc_dir):
        os.makedirs(noise_proc_dir)
        
    total_secs = total_hours*60*60
    total_samples = int(total_secs * fs)
    audio_length = int(audio_length*fs)
    SNR = np.linspace(snr_lower, snr_upper, int(total_snrlevels))
    cleanfilenames = glob.glob(os.path.join(clean_dir, audioformat))
    if not config["noise_types_excluded"]:
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
    else:
        filestoexclude = config["noise_types_excluded"].split(',')
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
        for i in range(len(filestoexclude)):
            noisefilenames = [fn for fn in noisefilenames if not os.path.basename(fn).startswith(filestoexclude[i])]
    
    filecounter = 0
    num_samples = 0
    
    while num_samples < total_samples:
        idx_s = np.random.randint(0, np.size(cleanfilenames))
        clean, fs = audioread(cleanfilenames[idx_s])
        
        if len(clean)>audio_length:
            clean = clean
        
        else:
            
            while len(clean)<=audio_length:
                idx_s = idx_s + 1
                if idx_s >= np.size(cleanfilenames)-1:
                    idx_s = np.random.randint(0, np.size(cleanfilenames)) 
                newclean, fs = audioread(cleanfilenames[idx_s])
                cleanconcat = np.append(clean, np.zeros(int(fs*silence_length)))
                clean = np.append(cleanconcat, newclean)
    
        idx_n = np.random.randint(0, np.size(noisefilenames))
        noise, fs = audioread(noisefilenames[idx_n])
        
        if len(noise)>=len(clean):
            noise = noise[0:len(clean)]
        
        else:
        
            while len(noise)<=len(clean):
                idx_n = idx_n + 1
                if idx_n >= np.size(noisefilenames)-1:
                    idx_n = np.random.randint(0, np.size(noisefilenames))
                newnoise, fs = audioread(noisefilenames[idx_n])
                noiseconcat = np.append(noise, np.zeros(int(fs*silence_length)))
                noise = np.append(noiseconcat, newnoise)
        noise = noise[0:len(clean)]
        filecounter = filecounter + 1
        
        annotation = ''
        for i in range(np.size(SNR)):
            clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=SNR[i])
            noisyfilename = 'noisy'+str(filecounter)+'_SNRdb_'+str(SNR[i])+'_clnsp'+str(filecounter)+'.wav'
            cleanfilename = 'clnsp'+str(filecounter)+'.wav'
            noisefilename = 'noisy'+str(filecounter)+'_SNRdb_'+str(SNR[i])+'.wav'
            noisypath = os.path.join(noisyspeech_dir, noisyfilename)
            cleanpath = os.path.join(clean_proc_dir, cleanfilename)
            noisepath = os.path.join(noise_proc_dir, noisefilename)
            annotation += os.path.abspath(noisypath) + ' ' + os.path.abspath(cleanpath) + '\n'
            audiowrite(noisy_snr, fs, noisypath, norm=False)
            audiowrite(clean_snr, fs, cleanpath, norm=False)
            audiowrite(noise_snr, fs, noisepath, norm=False)
            num_samples = num_samples + len(noisy_snr)
        
        out_file = os.path.join(os.path.dirname(__file__), 'out.txt')
        if config["out_file"]:
            out_file = config["out_file"]
        with open(out_file, 'w') as f:
            f.write(annotation)
        print(f'DONE')
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/noiser_config.yaml',
            help='YAML configuration file')
    return parser.parse_args()

def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__=="__main__":
    config = load_config(parse_args())
    
    main(config)