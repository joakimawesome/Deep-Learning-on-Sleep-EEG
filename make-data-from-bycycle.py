'''
Created on Thu Dec 1, 2022

@author: joakim
'''
import os
import tqdm
import numpy as np
import pandas as pd
from mne.io import read_raw_brainvision
from bycycle.group import compute_features_2d

def get_time_interval(annotation_file_name):
    """
    load the annotation file
    @author: ning
    @edit: joakim
    """
    # create time segments for cutting overlapping windows
    df_events               = pd.read_csv(annotation_file_name)

    # since we don't want to have too many "normal" data (labeled 0),
    # we cut off the last part of EEG when no particular events
    spindle_events          = df_events[df_events['Annotation'] == 'spindle']
    kcomplex_events         = df_events[df_events['Annotation'] == 'k-complex']
    stage_2_sleep_events    = df_events[df_events['Annotation'].apply(lambda x: np.logical_or(
                                            'Markon: 2' in x, 'Markoff: 2' in x))]

    tmin = 0
    # we only look at the data from when the first 2nd stage sleep started
    if len(stage_2_sleep_events) > 1:
        print('stage 2 sleep annotations are provided')
        tmin = np.min(stage_2_sleep_events['Onset'].values)
    
    # and we stop looking at the data when the last spindle, kcomplex, or 2nd stage stops,
    # whichever one happens the latest
    spindle_max = spindle_events['Onset'].values
    spindle_max = spindle_max if spindle_max.any() else np.zeros(1)
    kcomplex_max = kcomplex_events['Onset'].values
    kcomplex_max = kcomplex_max if kcomplex_max.any() else np.zeros(1)
    stage_2_max = stage_2_sleep_events['Onset'].values
    stage_2_max = stage_2_max if stage_2_max.any() else np.zero(1)
    
    tmax = np.max([spindle_max.max(), kcomplex_max.max() + 1, stage_2_max.max() + 1])
    return tmin, tmax

def get_onsets(annotation_file_name, event_type='spindle', tmin=0):
    '''
    retrieve event onsets
    '''
    df_events = pd.read_csv(annotation_file_name)
    spindle_events = df_events[df_events['Annotation'] == event_type]
    onsets = np.int32((spindle_events.Onset - tmin) * 1000) # in miliseconds
    return onsets


if __name__ == '__main__':
    EEG_dir             = './EEG'
    annotation_dir      = './annotations'
    bycycle_dir         = './cycles'

    if not os.path.exists(bycycle_dir):
        os.mkdir(bycycle_dir)

    info_for_all_subjects_dir = './get_data/data'
    df                        = pd.read_csv(os.path.join(info_for_all_subjects_dir, 'available_subjects.csv'))

    for (suj, day), df_sub in tqdm(df.groupby(['sub', 'day'])):

        fname_raw = './EEG/suj11_l2nap_day2.vhdr'
        channelList = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
        tmin, tmax = get_time_interval('./annotations/suj11_day2_annotations.txt')

        # Load data, crop for stage 2 sleep, pick the 6 channels, and 1-30 bandpass filter
        raw  = read_raw_brainvision(fname_raw, verbose=0)
        raw  = raw.crop(tmin, tmax)
        raw  = raw.pick_channels(channelList)
        filt = raw.load_data().filter(1, 30, verbose=0)

        sigs = filt.get_data()
        fs = 1000 # sample frequency
        f_range = (11, 16) # spindle range
        
        # Mnaually tuned thresholds (additional ML needed to optimize)
        thresholds = dict(amp_fraction_threshold=0.2,
                    amp_consistency_threshold=0.25,
                    period_consistency_threshold=0.5,
                    monotonicity_threshold=0.75,
                    min_n_cycles=3.)
        compute_kwargs = {'burst_method': 'cycles', 'threshold_kwargs': thresholds}

        # Compute cycle features for the 6 channels in parallel
        df_features_2d = compute_features_2d(sigs, fs, f_range, compute_features_kwargs=compute_kwargs, axis=None)

        onsets = get_onsets('./annotations/suj11_day2_annotations.txt', tmin=tmin)
        cycle_cols = ['sample_peak',
                      'sample_last_zerox_decay',
                      'sample_zerox_decay',
                      'sample_zerox_rise',
                      'sample_last_trough',
                      'sample_next_trough']

        for df_features in df_features_2d:
            arr = df_features[cycle_cols]
            mins = np.min(arr, axis=1)
            maxs = np.max(arr, axis=1)
            spindle_search = np.array([[1 if ((t < _min < t+1250) or (t < _max < t+1250)) else 0 \
                                        for t in onsets] for _min, _max in zip(mins, maxs)])
            in_spindle = np.any(spindle_search, axis=1)
            df_features['spindle'] = in_spindle.astype(int)

        for ch, df_features in zip(channelList, df_features_2d):
            df_features.to_csv(os.path.join(bycycle_dir,f'suj{suj}_day{day}_{ch}.csv'))