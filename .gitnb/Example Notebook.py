{'nbformat_minor': 1, 'nbformat': 4, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3}, 'file_extension': '.py', 'mimetype': 'text/x-python', 'name': 'python', 'nbconvert_exporter': 'python', 'pygments_lexer': 'ipython3', 'version': '3.6.4'}}}

###_MARKDONWBLOCK_###
#### Notes/ideas/todos

    - Data augmentation: https://github.com/drscotthawley/audio-classifier-keras-cnn/blob/master/augment_data.py
        - Implement augmentations for test data for averaging
    - MFCC instead of melspectro: https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    - Normalize datapoints individually? makes sense since melspectrograms are in dB, and we don't care so much about the dB level of the overall sample
    - Check the cropping, does it actually work..?
    - Explore labels and test data discrepancy from curated/noisy data
    - Ensemble
    - Clean up noisy data by training model on curated data and figuring out which noisy data is bad
    - Split each sample into several segments
    - Do variable length batching with CNN -> LSTM -> Dense model
###_MARKDONWBLOCK_###
#### Constants
###_CODEBLOCK_###
DO_PREPROCESS = True
DO_TRAIN = True

# data_dir = '../input/freesound-audio-tagging-2019/'
# checkpoint_dir = '../input/freesound-cnn-weights/'
# ensemble_checkpoint_dir = '../input/freesound-ensemble-weights/'
data_dir = '../input/'
checkpoint_dir = 'checkpoints/'
ensemble_checkpoint_dir = checkpoint_dir + 'ensemble/'

import os
for d in [data_dir, checkpoint_dir, ensemble_checkpoint_dir]:
    try:
        print(os.listdir(d))
        print()
    except:
        print("No such directory '{}'".format(d))
###_MARKDONWBLOCK_###
### Imports
###_CODEBLOCK_###
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import label_ranking_average_precision_score
from tqdm import tqdm_notebook as tqdm
import librosa
import os
import copy
import glob
import pickle

import warnings
warnings.filterwarnings('ignore')

import time
kernel_start_time = time.time()
###_MARKDONWBLOCK_###
### Helpers
###_MARKDONWBLOCK_###
#### Utils
###_CODEBLOCK_###
def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0, 
        scores[nonzero_weight_sample_indices, :], 
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap
###_MARKDONWBLOCK_###
#### Data
###_MARKDONWBLOCK_###
##### Preprocess
###_CODEBLOCK_###
def read_audio(file, config):
    
    data, _ = librosa.load(
        file, 
        sr=config['sampling_rate'],
        res_type='kaiser_fast')
    
    if len(data) > 0:
        data, _ = librosa.effects.trim(data, top_db=60)
        
    return data
    

def extract_mel_spec(data, config):
    
    mel_spec = librosa.feature.melspectrogram(
        data.astype(float), 
        n_fft=config['n_fft'], 
        hop_length=config['hop_length'], 
        n_mels=config['n_mels'], 
        sr=config['sampling_rate'], 
        power=1.0, 
        fmin=config['fmin'], 
        fmax=config['fmax'])

    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def pad_features(mel_spec_db, seq_len):
    
    if len(mel_spec_db) > seq_len:
        features = mel_spec_db[:seq_len]
    else:
        features = np.zeros((seq_len, mel_spec_db.shape[1]))
        features[:len(mel_spec_db)] = mel_spec_db
        
    return features


def create_mel_specs(fnames, data_dir, config, to_disk=True, out_dir=None):

    if to_disk:
        assert out_dir is not None
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        all_features = np.zeros((len(fnames), config['seq_len'], config['n_mels']))
    
    pbar = tqdm(total=len(fnames))
    for i, fname in enumerate(fnames):
        
        audio_data = read_audio(data_dir + fname, config)
        mods = augment_data(audio_data, sr=config['sampling_rate'], n_augment=config['n_augment'])
        for mod_n, augmented_audio_data in enumerate(mods):
        
            mel_spec_db = extract_mel_spec(augmented_audio_data, config)

            mel_spec_db = (mel_spec_db - config['scale_mean']) / config['scale_std']
            mel_spec_db = mel_spec_db.transpose(1, 0) # (time, features)

            if to_disk:
                fname_out = fname.replace('.wav', '_{}.pkl'.format(mod_n))
                with open(out_dir + fname_out, 'wb') as f:
                    pickle.dump(mel_spec_db, f)
            else:
                
                features = pad_features(mel_spec_db, config['seq_len'])
                all_features[i] = features
                # Do no augmentations at test time
                break

        pbar.update(1)

    if not to_disk:
        return all_features
###_MARKDONWBLOCK_###
##### Augment
###_CODEBLOCK_###
## CREDIT: https://github.com/drscotthawley/audio-classifier-keras-cnn/blob/master/augment_data.py

from random import getrandbits


def random_onoff():                # randomly turns on or off
    return bool(getrandbits(1))


# returns a list of augmented audio data, stereo or mono
def augment_data(y, sr, n_augment=0, allow_speedandpitch=True, allow_pitch=True,
    allow_speed=True, allow_dyn=True, allow_noise=True, allow_timeshift=True, tab="", verbose=False):

    mods = [y]                  # always returns the original as element zero
    length = y.shape[0]

    for i in range(n_augment):
        if verbose: print(tab+"augment_data: ",i+1,"of",n_augment)
        y_mod = y
        count_changes = 0

        # change speed and pitch together
        if (allow_speedandpitch) and random_onoff():   
            length_change = np.random.uniform(low=0.9,high=1.1)
            speed_fac = 1.0  / length_change
            if verbose: print(tab+"    resample length_change = ",length_change)
            tmp = np.interp(np.arange(0,len(y),speed_fac),np.arange(0,len(y)),y)
            #tmp = resample(y,int(length*lengt_fac))    # signal.resample is too slow
            minlen = min( y.shape[0], tmp.shape[0])     # keep same length as original; 
            y_mod *= 0                                    # pad with zeros 
            y_mod[0:minlen] = tmp[0:minlen]
            count_changes += 1

        # change pitch (w/o speed)
        if (allow_pitch) and random_onoff():   
            bins_per_octave = 24        # pitch increments are quarter-steps
            pitch_pm = 4                                # +/- this many quarter steps
            pitch_change =  pitch_pm * 2*(np.random.uniform()-0.5)   
            if verbose: print(tab+"    pitch_change = ",pitch_change)
            y_mod = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
            count_changes += 1

        # change speed (w/o pitch), 
        if (allow_speed) and random_onoff():   
            speed_change = np.random.uniform(low=0.9,high=1.1)
            if verbose: print(tab+"    speed_change = ",speed_change)
            tmp = librosa.effects.time_stretch(y_mod, speed_change)
            minlen = min( y.shape[0], tmp.shape[0])        # keep same length as original; 
            y_mod *= 0                                    # pad with zeros 
            y_mod[0:minlen] = tmp[0:minlen]
            count_changes += 1

        # change dynamic range
        if (allow_dyn) and random_onoff():  
            dyn_change = np.random.uniform(low=0.5,high=1.1)  # change amplitude
            if verbose: print(tab+"    dyn_change = ",dyn_change)
            y_mod = y_mod * dyn_change
            count_changes += 1

        # add noise
        if (allow_noise) and random_onoff():  
            noise_amp = 0.005*np.random.uniform()*np.amax(y)  
            if random_onoff():
                if verbose: print(tab+"    gaussian noise_amp = ",noise_amp)
                y_mod +=  noise_amp * np.random.normal(size=length)  
            else:
                if verbose: print(tab+"    uniform noise_amp = ",noise_amp)
                y_mod +=  noise_amp * np.random.normal(size=length)  
            count_changes += 1

        # shift in time forwards or backwards
        if (allow_timeshift) and random_onoff():
            timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
            if verbose: print(tab+"    timeshift_fac = ",timeshift_fac)
            start = int(length * timeshift_fac)
            if (start > 0):
                y_mod = np.pad(y_mod,(start,0),mode='constant')[0:y_mod.shape[0]]
            else:
                y_mod = np.pad(y_mod,(0,-start),mode='constant')[0:y_mod.shape[0]]
            count_changes += 1

        # last-ditch effort to make sure we made a change (recursive/sloppy, but...works)
        if (0 == count_changes):
            if verbose: print("No changes made to signal, trying again")
            mods.append(  augment_data(y, sr, n_augment = 1, tab="      ")[1] )
        else:
            mods.append(y_mod)

    return mods
###_MARKDONWBLOCK_###
##### Containers
###_CODEBLOCK_###
class MelSpecDatasetPreprocessed(Dataset):
    
    def __init__(self, features_info, config, train=True, from_disk=True, data_dir=None, labels=None):
        
        if from_disk:
            assert data_dir is not None, "Must supply data_dir"
        
        self.features_info = features_info
        self.data_dir = data_dir
        self.config = config
        self.labels = labels
        self.from_disk = from_disk
        self.train = train
    
    def __len__(self):
            
        return len(self.features_info)
    
    def __getitem__(self, idx):
        
        if self.from_disk:
            random_augmentation_idx = np.random.randint(0, self.config['n_augment'] + 1)
            
            pkl_fname = self.features_info[idx].replace(
                '.wav', '_{}.pkl'.format(random_augmentation_idx))
            with open(self.data_dir + pkl_fname, 'rb') as f:
                mel_spec_db = pickle.load(f)
                
            features = pad_features(mel_spec_db, config['seq_len'])
            
        else:
            features = self.features_info[idx]
            
        if self.train:
            if self.config['crop_ratio'] > 0.0:
                crop_length = int(len(features) * (1 - self.config['crop_ratio']))
                random_start = np.random.randint(
                    0, int(len(features) * self.config['crop_ratio']))
                end = random_start + crop_length
                features_crop = features[random_start : end]
                features = np.zeros_like(features)
                features[0 : crop_length] = features_crop
                
        features = torch.tensor(features, dtype=torch.float32)
        
        if self.labels is not None:
            
            if self.config['crop_ratio'] > 0.0 and features.sum() == 0.0:
                labels = torch.zeros(self.config['num_classes'])
            else:
                labels = torch.tensor(self.labels[idx], dtype=torch.float32)
                
            return {
                'features' : features,
                'labels' : labels,
            }

        else:
            return {
                'features' : features,
            }
        

class MelSpecDatasetUnprocessed(Dataset):
    
    def __init__(self, features_info, config, data_dir=None, train=True, labels=None):
        
        self.features_info = features_info
        self.data_dir = data_dir
        self.config = config
        self.labels = labels
        self.train = train
    
    def __len__(self):
            
        return len(self.features_info)
    
    def __getitem__(self, idx):
        
        fname = self.data_dir + self.features_info[idx]
        audio_data = read_audio(data_dir + fname, self.config)
        if self.train:
            audio_data = augment_data(audio_data, sr=self.config['sampling_rate'], n_augment=1)[-1]

        mel_spec_db = extract_mel_spec(audio_data, self.config)
        mel_spec_db = (mel_spec_db - self.config['scale_mean']) / self.config['scale_std']
        mel_spec_db = mel_spec_db.transpose(1, 0) # (time, features)

        features = pad_features(mel_spec_db, self.config['seq_len'])
            
        if self.train:
            if self.config['crop_ratio'] > 0.0:
                crop_length = int(len(features) * (1 - self.config['crop_ratio']))
                random_start = np.random.randint(
                    0, int(len(features) * self.config['crop_ratio']))
                end = random_start + crop_length
                features_crop = features[random_start : end]
                features = np.zeros_like(features)
                features[0 : crop_length] = features_crop
                
        features = torch.tensor(features, dtype=torch.float32)
        
        if self.labels is not None:
            
            if self.config['crop_ratio'] > 0.0 and features.sum() == 0.0:
                labels = torch.zeros(self.config['num_classes'])
            else:
                labels = torch.tensor(self.labels[idx], dtype=torch.float32)
                
            return {
                'features' : features,
                'labels' : labels,
            }

        else:
            return {
                'features' : features,
            }
        
        
class SubsetRandomOversampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, 
            oversample_subset, oversample_factor=1):
        
        ## OBS: Assumes that the oversample_subset is the first dataset in data_source

        self.data_source = data_source
        self.oversample_indices = [idx for idx in range(len(oversample_subset))] * oversample_factor
        self.undersample_indices = [idx for idx in range(len(oversample_subset), len(data_source))]
        self.sample_indices = np.array(self.oversample_indices + self.undersample_indices)

    def __iter__(self):
        
        np.random.shuffle(self.sample_indices)
        return iter(self.sample_indices)

    def __len__(self):
        return len(self.sample_indices)
###_MARKDONWBLOCK_###
#### Models
###_CODEBLOCK_###
class CNN(nn.Module):
    
    def __init__(self, config):
        
        super(CNN, self).__init__()
        
        height = config['seq_len']
        width = config['n_mels']
        n_filters = [1] + config['n_filters']
        
        ## Construct the convolutional layers
        convs = []
        print("Intermediate image sizes:")
        print((height, width))
        
        for i, f_size in enumerate(config['filter_sizes']):
            
            conv = []
            
            ## Add the conv
            if config['pad_convs']:
                padding = (int(np.ceil((f_size[0] - 1) / 2)), 
                           int(np.ceil((f_size[1] - 1) / 2)))
            else:
                padding = 0
            
            conv.append(nn.Conv2d(n_filters[i], n_filters[i+1], f_size, 
                              stride=config['strides'][i],
                              padding=padding))
            
            ## Reduce img size for conv
            height = (height + 2 * padding[0] - f_size[0]) // config['strides'][i][0] + 1
            width = (width + 2 * padding[1] - f_size[1]) // config['strides'][i][1] + 1
            
            ## Add pool
            pool = config['pools'][i]
            if pool:
                conv.append(nn.MaxPool2d(pool))
                ## Reduce img size for max pooling
                height = (height - (pool - 1) - 1) // pool + 1
                width = (width - (pool - 1) - 1) // pool + 1
            
            ## Add ReLU
            conv.append(nn.ReLU())
            
            ## Add Batch norm
            if i%config['batch_norm_interval'] == config['batch_norm_interval']-1:
                conv.append(nn.BatchNorm2d(n_filters[i+1]))
            
            ## Append the layer
            conv = nn.Sequential(*conv)
            convs.append(conv)
                
            print((height, width))
            
        self.convs = nn.Sequential(*convs)
        conv_output_size = config['n_filters'][-1] * height * width
        self.dense = nn.Linear(conv_output_size, config['dense_size'])
        self.dropout = nn.Dropout(config['dropout'])
        
        self.classifier = nn.Linear(
                config['dense_size'], 
                config['num_classes'])
        
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, x, labels=None):
        
        x = x.unsqueeze(1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.dense(x)))
        logits = self.classifier(x)
        
        if labels is None:
            return logits
        else:
            return logits, self.criterion(logits, labels)
###_MARKDONWBLOCK_###
#### Model wrapper
###_CODEBLOCK_###
class ModelWrapper:
    
    def __init__(self, config=None, pretrained_path=None):
        
        if config is not None:
            self.config = config.copy()
        if pretrained_path is not None:
            self.config = torch.load(pretrained_path + '_config.pth')

        self.config['use_cuda'] = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.config['use_cuda'] else "cpu")
        
        self.init_model(pretrained_path=pretrained_path)
        
    def init_model(self, pretrained_path=None):
        
        model_choice = self.config['model_choice']
    
        if model_choice == 0:
            self.net = CNN(self.config).to(self.device)
            self.weight_initialization(init='kaiming')
            
        else:
            print("ERROR: NO MODEL SELECTED")
            return False
        
        self.init_optimizer()

        if pretrained_path is not None:
            self.load_state(pretrained_path)
        
        return True
    
    def init_optimizer(self):

        optim_choice = self.config['optim_choice']

        if optim_choice == 0:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'])
            self.lr_scheduler = None
            
        elif optim_choice == 1:
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.00001, momentum=0.9)
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
        else:
            print("ERROR: NO OPTIMIZER SELECTED")
            return False
            
        return True
    
    def weight_initialization(self, init='xavier', keywords=['']):
        for name, param in self.net.named_parameters():
            if 'weight' in name and any(k in name for k in keywords) \
                                and len(param.shape) > 1:
                if init == 'xavier':
                    torch.nn.init.xavier_normal_(param)
                elif init == 'kaiming':
                    torch.nn.init.kaiming_normal_(param)

    def get_summary(self):

        param_counts = [[n, p.numel()] for n, p in self.net.named_parameters()]
        params_summary = pd.DataFrame(param_counts, columns=['name', '# params'])
        num_params = params_summary['# params'].sum()
        params_summary['# params'] = list(map("{:,}".format, params_summary['# params']))

        return params_summary, num_params

    def update_config(self, changes):

        for param in changes:
            if param not in self.config:
                warning_str = "'{}' not in config.".format(param)
                warnings.warn(warning_str)
            
        self.config.update(changes)

        if 'lr' in changes:
            self.update_learning_rate(changes['lr'])

    def update_learning_rate(self, lr):

        self.config['lr'] = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train(self, train_loader, valid_loader, verbose=2):
    
        self.net.train()

        if self.config['revert_after_training']:
            best_model_params = copy.deepcopy(self.net.state_dict())
            best_optimizer_params = copy.deepcopy(self.optimizer.state_dict())
            best_metrics, best_reports = self.evaluate(valid_loader)
            best_eval_metric = self.get_eval_metric_as_loss(best_metrics)
        else:
            best_metrics = {}
            best_reports = {}
            best_eval_metric = np.inf

        global_step = 0
        smooth_loss = 0.0

        eval_step = max(1, int(len(train_loader) 
            * 1.0 / self.config['eval_steps_per_epoch']))

        best_step = 1 # Dont stop before this many steps + patience
        patience = self.config['patience']
        if patience is not None:
            patience = int(eval_step * self.config['patience'])
            best_step = patience if not self.config['revert_after_training'] else 1

        for e in range(1, self.config['num_epochs'] + 1):

            if patience and global_step >= best_step + patience and best_eval_metric != np.inf:
                break
                    
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            if verbose >= 1:
                print("---------- EPOCH {}/{} ----------\n".format(
                    e, self.config['num_epochs']))

            for i, batch in enumerate(train_loader):

                if patience and global_step >= best_step + patience and best_eval_metric != np.inf:
                    break

                output, loss = self.do_forward_pass(batch)
                
                self.do_backward_pass(loss)

                batch_loss = loss.detach().cpu().item()
                if global_step == 0:
                    smooth_loss = batch_loss
                else:
                    smooth_loss = smooth_loss * 0.99 + batch_loss * 0.01

                global_step += 1

                if global_step % eval_step == 0 and (not patience or global_step > patience or best_eval_metric != np.inf):

                    valid_metrics, valid_reports = self.evaluate(valid_loader)
                    valid_eval_metric = self.get_eval_metric_as_loss(valid_metrics)

                    if valid_eval_metric < best_eval_metric:
                        if self.config['revert_after_training']:
                            best_model_params = copy.deepcopy(self.net.state_dict())
                            best_optimizer_params = copy.deepcopy(self.optimizer.state_dict())
                        best_eval_metric = valid_eval_metric
                        best_metrics = valid_metrics
                        best_reports = valid_reports
                        best_step = global_step
                        if verbose >= 2:
                            print("New best!")
                    
                    if verbose >= 2:
                        print("Step: {}/{}".format(i+1, len(train_loader)))
                        print("Total steps: {}".format(global_step))
                        print("Training loss (smooth): {:.5f}".format(smooth_loss))
                        self.print_evals(valid_metrics, valid_reports)
                        if patience is not None:
                            print("Best {} so far: {:.5f} ({} evals until stopping)".format(
                                self.config['eval_metric'], best_metrics[self.config['eval_metric']], 
                                (patience + best_step - global_step) // eval_step))

                        if self.config['use_cuda']:
                            max_allocated = torch.cuda.max_memory_allocated() / 10**9
                            print("Maximum GPU consumption so far: {:.3f} [GB]".format(max_allocated))
                            
                        print()
                    
        if verbose >= 1:
            self.print_evals(best_metrics, best_reports)
            print("At step:", best_step)
            
        if self.config['revert_after_training']:
            self.net.load_state_dict(best_model_params)
            self.optimizer.load_state_dict(best_optimizer_params)
            
    def do_forward_pass(self, batch):

        features = batch['features'].to(self.device)
        labels = batch['labels'].to(self.device) if 'labels' in batch else None

        return self.net(features, labels=labels)
    
    def do_backward_pass(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        if self.config['clip']:
            nn.utils.clip_grad_norm_(self.net.parameters(), 
                                     self.config['clip'])
        self.optimizer.step()
    
    def get_eval_metric_as_loss(self, evals):
        
        eval_metric = self.config['eval_metric']
        if eval_metric == 'loss':
            return evals['loss']
        elif eval_metric == 'lwlrap':
            return -evals['lwlrap']
        else:
            raise KeyError("Unknown eval metric {}.".format(eval_metric))

    def evaluate(self, loader):

        loss = 0.0
        tot_sequences = 0
        labels_all = []
        probs_all = []

        self.net.eval() # Put model in evaluation mode

        with torch.no_grad():

            for batch in loader:

                ## Get data
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                batch_size = len(features)
                tot_sequences += batch_size

                ## Get output
                logits, batch_loss = self.net(features, labels=labels)
                loss += batch_loss.item() * batch_size

                labels_all.extend(labels.cpu().numpy())
                probs_all.extend(torch.sigmoid(logits).cpu().numpy())

        self.net.train() # Put model back in training mode

        labels_all = np.array(labels_all)
        probs_all = np.array(probs_all)
        
        lwlrap = calculate_overall_lwlrap_sklearn(labels_all, probs_all)

        metrics = {
            'loss' : loss / tot_sequences,
            'lwlrap' : lwlrap,
        }
        
        reports = {}

        return metrics, reports
    
    def print_evals(self, metrics, reports, dataset='validation'):
        
        for name, metric in metrics.items():
            print("{} {}: {:.5f}".format(dataset.capitalize(), name, metric))
            
        for name, report in reports.items():
            print("{} {}:".format(dataset.capitalize(), name))
            print(report)
    
    def predict(self, loader):

        self.net.eval() # Put model in evaluation mode
        preds = []

        with torch.no_grad():

            for batch in loader:

                ## Get data
                features = batch['features'].to(self.device)

                ## Get output
                output = self.net(features)

                batch_preds = torch.sigmoid(output).cpu().numpy()
                preds.extend(batch_preds)

        return np.array(preds)

    def save_state(self, folder, name):

        if not os.path.exists(folder):
            os.makedirs(folder)

        model_path = folder + name
        torch.save(self.net.state_dict(), model_path + '.pth')
        torch.save(self.optimizer.state_dict(), model_path + '_optim.pth')
        torch.save(self.config, model_path + '_config.pth')

    def load_state(self, path):

        self.config = torch.load(path + '_config.pth')
        self.config['use_cuda'] = torch.cuda.is_available()
        
        if self.config['use_cuda']:
            state_dict = torch.load(path + '.pth')
        else:
            state_dict = torch.load(path + '.pth', map_location='cpu')

        self.net.load_state_dict(state_dict)
        optimizer_state_dict = torch.load(path + '_optim.pth')
        self.optimizer.load_state_dict(optimizer_state_dict)
###_MARKDONWBLOCK_###
#### Ensemble wrapper
###_CODEBLOCK_###
class EnsembleWrapper():
    
    def __init__(self, config, pretrained_path=None):
        
        self.config = config.copy()
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        
        if pretrained_path is not None:
            self.load_models(pretrained_path)

    def update_config(self, changes):

        for param in changes:
            if param not in self.config:
                warning_str = "'{}'' not in config.".format(param)
                warnings.warn(warning_str)

        self.config.update(changes)
        
    def evaluate(self, loader):

        n_models = len(self.models)
        labels = []
        probs = []
        tot_sequences = 0

        for model in self.models: model.net.eval()
            
        with torch.no_grad():

            for batch in loader:

                batch_size = len(batch['features'])
                tot_sequences += batch_size

                batch_probs = torch.zeros((batch_size, 
                    self.config['output_size'])).to(self.device)

                combined_output = torch.zeros((batch_size,
                    self.config['output_size'])).to(self.device)

                for i_m, model in enumerate(self.models):
                    ## Get output of each model
                    output, loss = model.do_forward_pass(batch)
                    batch_probs += torch.sigmoid(output)
                    combined_output += output

                combined_output /= n_models
                loss += self.criterion(combined_output, labels).item() * batch_size
                
                batch_probs /= n_models
                probs.extend(batch_probs.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())

        for model in self.models: model.net.train()
                
        metrics = {
            'loss' : loss / tot_sequences,
            'lwlrap' : calculate_overall_lwlrap_sklearn(labels, probs)
        }
        reports = {}
        
        return metrics, reports

    def predict(self, loader):
            
        n_models = len(self.models)
        probs = []

        for model in self.models: model.net.eval()

        with torch.no_grad():

            for batch in loader:

                for i, model in enumerate(self.models):
                    
                    if 'labels' in batch:
                        output, loss = model.do_forward_pass(batch)
                    else:
                        output = model.do_forward_pass(batch)
                    
                    if i == 0:
                        batch_probs = torch.sigmoid(output)
                    else:
                        batch_probs += torch.sigmoid(output)

                batch_probs /= n_models
                probs.extend(batch_probs.cpu().numpy())

        return np.array(probs)
    
    def train(self, fnames_curated, labels_curated,
              fnames_noisy, labels_noisy,
              stage_configs,
              n_models=10, verbose=1):
      
        kfold = KFold(n_splits=n_models, random_state=1, shuffle=True)

        avg_loss, avg_acc = 0., 0.
        avg_cfm = np.zeros((3, 3))

        self.models = []
        noisy_indices = [i for i in range(len(fnames_noisy))]

        for i, (train_index, valid_index) in enumerate(kfold.split(fnames_curated)):

            if verbose >= 1:
                print('============>', 'Training Model {}...'.format(i + 1))

            ## Split data
            fnames_curated_train, fnames_curated_valid = (
                fnames_curated[train_index], fnames_curated[valid_index]
            )
            
            labels_curated_train, labels_curated_valid = (
                labels_curated[train_index], labels_curated[valid_index]
            )
            
            train_data_curated = MelSpecDatasetPreprocessed(
                fnames_curated_train, config, 
                train=True,
                data_dir=data_dir+'train_curated_preprocessed/', 
                labels=labels_curated_train)
            
            noisy_subset_indices = np.random.choice(
                noisy_indices, size=int(len(noisy_indices)*0.8), replace=False)
            
            train_data_noisy = MelSpecDatasetPreprocessed(
                fnames_noisy[noisy_subset_indices], 
                config, 
                train=True,
                data_dir=data_dir+'train_noisy_preprocessed/', 
                labels=labels_noisy[noisy_subset_indices])

            train_data = train_data_curated + train_data_noisy

            sampler = SubsetRandomOversampler(
                 train_data, train_data_curated, oversample_factor=1)

            train_loader = DataLoader(train_data, 
                                      sampler=sampler,
                                      #shuffle=True,
                                      num_workers=4,
                                      batch_size=config['batch_size'])

            valid_data = MelSpecDatasetPreprocessed(
                fnames_curated_valid, config, 
                train=False,
                data_dir=data_dir+'train_curated_preprocessed/', 
                labels=labels_curated_valid)

            valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)

            ## Initiate the model
            model = ModelWrapper(self.config)

            ## Train the model
            for stage_i, config_change in enumerate(stage_configs):
                
                if stage_i == 1:
                    
                    train_loader = DataLoader(
                        train_data_curated, 
                        shuffle=True,
                        num_workers=4,
                        batch_size=config['batch_size']
                    )
                
                model.update_config(config_change)
                model.train(train_loader, valid_loader, verbose=verbose)

            self.models.append(model)

    def save_models(self, checkpoint_dir):
        
        for i, model in enumerate(self.models):
            
            name = 'model_' + str(i)
            model.save_state(checkpoint_dir, name)
            
    def load_models(self, checkpoint_dir, n_models=10):
        
        self.models = []
        
        for i in range(n_models):
            
            checkpoint_path = checkpoint_dir + 'model_' + str(i)
            model = ModelWrapper(pretrained_path=checkpoint_path)
            self.models.append(model)
###_MARKDONWBLOCK_###
### Load data
###_CODEBLOCK_###
data_curated = pd.read_csv(data_dir + 'train_curated.csv')
data_noisy = pd.read_csv(data_dir + 'train_noisy.csv')

## Remove trash examples
corrupt_examples = [
    '77b925c2.wav',
]
data_curated = data_curated[~data_curated['fname'].isin(corrupt_examples)]

sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')
all_labels = sorted(sample_submission.columns[1:].tolist())
label_to_idx = {l : i for i, l in enumerate(all_labels)}

print("{} curated data.".format(len(data_curated)))
print("{} noisy data.".format(len(data_noisy)))
print("{} labels.".format(len(all_labels)))
###_MARKDONWBLOCK_###
### Hyperparameters
###_CODEBLOCK_###
config = dict(
    ## Melspectrogram hyperparameters
    n_fft = 2048,
    hop_length = 500,
    n_mels = 64,
    fmin = 40,
    fmax = 18000,
    sampling_rate = 32000,
    
    ## Data processing hyperparameters
    use_cuda = torch.cuda.is_available(),
    num_classes = len(all_labels),
    max_seconds = 30,
    seq_type = 'split', # split or pad sequences
    n_augment = 10,
    scale_mean = 80,
    scale_std = 12,
    crop_ratio = 0.8,
    
    ## Training hyperparameters
    num_epochs = 10,
    batch_size = 32,
    lr = 0.001,
    clip = 0.1, # Gradient clipping
    eval_steps_per_epoch = 2, 
    patience = 10, # measured in nr of eval steps
    revert_after_training = True, # If true, reverts model parameters after training to best found during early stopping
    model_choice = 0,
    optim_choice = 0,
    eval_metric = 'loss',
)

config['seq_len'] = (config['sampling_rate'] * config['max_seconds']) // config['hop_length']

print("seq length:", config['seq_len'])
device = torch.device('cuda' if config['use_cuda'] else 'cpu')
device
###_MARKDONWBLOCK_###
#### Find mean values for scaling
###_CODEBLOCK_###
def get_mean_std(vals):
    
    return vals.mean(axis=1), vals.std(axis=1)
###_CODEBLOCK_###
means_stds = data_curated['fname'][0:20].apply(
    lambda f: get_mean_std(
        extract_mel_spec(
            read_audio(data_dir + 'train_curated/' + f, config), 
            config
        )
    )
)
###_CODEBLOCK_###
means = np.zeros(64)
stds = np.zeros(64)
for m, s in means_stds:
    
    means += m
    stds += s
    
means /= len(means_stds)
stds /= len(means_stds)
###_CODEBLOCK_###
stds.max(), means.min()
###_MARKDONWBLOCK_###
To approximately scale values to be normal distributed, we can subtract -80 and divide by 12. A better approach might be to scale the channels individually, but they are quite close in magnitude so it may not be necessary.
###_MARKDONWBLOCK_###
### Preprocess data
###_CODEBLOCK_###
def labels_to_np(labels):
    
    labels_np = np.zeros((len(labels), len(all_labels)), dtype=np.long)
    for i, labels_str in enumerate(labels):
        for l in labels_str.split(','):
            j = label_to_idx[l]
            labels_np[i, j] = 1
            
    return labels_np
###_CODEBLOCK_###
fnames_curated = data_curated['fname'].values
labels_curated = labels_to_np(data_curated['labels'].values)

fnames_noisy = data_noisy['fname'].values
labels_noisy = labels_to_np(data_noisy['labels'].values)
###_CODEBLOCK_###
if DO_PREPROCESS:
    
    import shutil
    
    in_folder = data_dir + 'train_curated/'
    out_folder = data_dir + 'train_curated_preprocessed/'
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    create_mel_specs(
        fnames_curated, in_folder, config, 
        to_disk=True, out_dir=out_folder)
###_CODEBLOCK_###
if DO_PREPROCESS:
    
    in_folder = data_dir + 'train_noisy/'
    out_folder = data_dir + 'train_noisy_preprocessed/'
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    create_mel_specs(
        fnames_noisy, in_folder, config, 
        to_disk=True, out_dir=out_folder)
###_MARKDONWBLOCK_###
### Create train/valid dataloaders
###_CODEBLOCK_###
if DO_TRAIN:
    
    train_fnames, valid_fnames, train_labels, valid_labels = train_test_split(
        fnames_curated, labels_curated,
        test_size=0.2, random_state=1,
    )

    train_data_curated = MelSpecDatasetUnprocessed(
        train_fnames, config, 
        train=True,
#         data_dir=data_dir+'train_curated_preprocessed/', 
        data_dir=data_dir+'train_curated/', 
        labels=train_labels)

    train_data_noisy = MelSpecDatasetUnprocessed(
        fnames_noisy, config, 
        train=True,
#         data_dir=data_dir+'train_noisy_preprocessed/', 
        data_dir=data_dir+'train_noisy/', 
        labels=labels_noisy)

    train_data = train_data_curated + train_data_noisy

    sampler = SubsetRandomOversampler(
         train_data, train_data_curated, oversample_factor=1)

    train_loader = DataLoader(train_data, 
                              sampler=sampler,
                              #shuffle=True,
                              num_workers=4,
                              batch_size=config['batch_size'])

    valid_data = MelSpecDatasetUnprocessed(
        valid_fnames, config, 
        train=False,
#         data_dir=data_dir+'train_curated_preprocessed/', 
        data_dir=data_dir+'train_curated/', 
        labels=valid_labels)

    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)
###_MARKDONWBLOCK_###
### Train model
###_CODEBLOCK_###
model_hyperparameters = dict(
    pad_convs = True,
    n_filters = [64, 128, 128, 256, 512],
    filter_sizes = [(20, 7), (10, 5), (5, 3), (3, 2), (3, 2)],
    strides = [(5, 1), (3, 1), (2, 1), (1, 1), (1, 1)],
    pools = [2, 2, 2, 2, 2, 2],
    batch_norm_interval = 1,
    dense_size = 1024,
    dropout = 0.4,
)
config.update(model_hyperparameters)
###_CODEBLOCK_###
model = ModelWrapper(config=config)
###_CODEBLOCK_###
summary, n_params = model.get_summary()
print("{:,} total parameters".format(n_params))
summary
###_CODEBLOCK_###
if DO_TRAIN:
    
    config_changes = dict(
        num_epochs = 100,
        revert_after_training = True,
        eval_steps_per_epoch = 2,
        patience = 30,
        clip = None,
        lr = 0.001,
    )
    model.update_config(config_changes)

    model.train(train_loader, valid_loader, verbose=2)
    print("Preperatory training finished!")
###_CODEBLOCK_###
if DO_TRAIN:
    ## Initial training with noisy dataset
    
    config_changes = dict(
        num_epochs = 20,
        revert_after_training = True,
        eval_steps_per_epoch = 6,
        patience = 20,
        clip = None,
        lr = 0.001,
    )
    model.update_config(config_changes)

    model.train(train_loader, valid_loader, verbose=2)
    print("Preperatory training finished!")
###_CODEBLOCK_###
if DO_TRAIN:
    ## Training with only curated data
    config_changes = dict(
        num_epochs = 100,
        eval_steps_per_epoch = 2,
    )
    model.update_config(config_changes)
    
    train_loader = DataLoader(train_data_curated, 
                          shuffle=True,
                          num_workers=4,
                          batch_size=config['batch_size'])
    
    model.train(train_loader, valid_loader, verbose=2)
    print("Preperatory training finished!")
###_CODEBLOCK_###
if DO_TRAIN:
    ## Full training with lower learning rate and some gradient clipping
    config_changes = dict(
        clip = 0.01,
        lr = 0.0001,
    )
    model.update_config(config_changes)

    model.train(train_loader, valid_loader)
###_CODEBLOCK_###
if DO_TRAIN:
    ## Mega deep training
    config_changes = dict(
        clip = 0.01,
        lr = 0.00001,
    )
    model.update_config(config_changes)

    model.train(train_loader, valid_loader)
###_CODEBLOCK_###
if DO_TRAIN:
    model.save_state(checkpoint_dir, 'model_0')
###_MARKDONWBLOCK_###
### Train ensemble
###_CODEBLOCK_###
model_hyperparameters = dict(
    pad_convs = True,
    n_filters = [64, 128, 128, 256, 512],
    filter_sizes = [(20, 7), (10, 5), (5, 3), (3, 2), (3, 2)],
    strides = [(5, 1), (3, 1), (2, 1), (1, 1), (1, 1)],
    pools = [2, 2, 2, 2, 2, 2],
    batch_norm_interval = 1,
    dense_size = 1024,
    dropout = 0.4,
)
config.update(model_hyperparameters)
###_CODEBLOCK_###
if DO_TRAIN:
    ensemble = EnsembleWrapper(config=config)
###_CODEBLOCK_###
if DO_TRAIN:

    stage_configs = [
        dict(
            num_epochs = 20,
            revert_after_training = True,
            eval_steps_per_epoch = 6,
            patience = 20,
            clip = None,
            lr = 0.001,
        ),
        dict(
            num_epochs = 100,
            eval_steps_per_epoch = 2,
        ),
        dict(
            clip = 0.01,
            lr = 0.0001,
        ),
        dict(
            lr = 0.00001,
        )
    ]
    ensemble.train(fnames_curated, labels_curated, fnames_noisy, labels_noisy, stage_configs)
###_CODEBLOCK_###
if DO_TRAIN:
    ensemble.save_models(ensemble_checkpoint_dir)
###_MARKDONWBLOCK_###
### Predict test set
###_CODEBLOCK_###
#model = ModelWrapper(pretrained_path=checkpoint_dir + 'model_0')
model = EnsembleWrapper(config, pretrained_path=ensemble_checkpoint_dir)
###_CODEBLOCK_###
test_dir = data_dir + 'test/'
test_fnames = os.listdir(test_dir)
###_CODEBLOCK_###
test_features = create_mel_specs(test_fnames, test_dir, config, to_disk=False)
###_CODEBLOCK_###
test_data = MelSpecDatasetPreprocessed(test_features, config, train=False, from_disk=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
###_CODEBLOCK_###
predictions = model.predict(test_loader)
###_CODEBLOCK_###
predictions_df = pd.DataFrame(predictions, columns=all_labels)
label_columns_sorted = sorted(predictions_df.columns)
predictions_df['fname'] = test_fnames
predictions_df = predictions_df[['fname'] + label_columns_sorted]
predictions_df.head()
###_CODEBLOCK_###
predictions_df.to_csv('submission.csv', index=None)