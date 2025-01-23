## first-level GLM analysis for theory of mind (ToM) and social interactions with online participants' ratings 
# Author: Zizhuang Miao

# Main regressors:
# rating: a box-car regressor for the rating period in every trial (the last 8 seconds)
# ToM: a float number from -1 to 1, representing the median of online participants' ratings of how much they were using ToM at each moment (-1 is not at all and 1 is all the time)
# SInt: a float number from -1 to 1, representing the median of online participants' ratings of social interactions at each moment (-1 is not at all and 1 is definitely yes)
# all the beta maps from the main regressors will be saved

# Nuisance regressors:
# 24 Motion parameters
# csf mean activity
# discrete cosine basis functions (as a high pass filter at 1/128 Hz)
# indicator function of each spike (defined as outside 3*SD from the global mean or from the previous volume)
# indicator function of each invalid time point (see analysis of behavioral data)

# BOLD data were first spatial smoothed using a Gaussian kernel (full-width-at-half-maximum = 6mm) before being regressed
# we will directly use the smoothed BOLD data

import pandas as pd
import numpy as np
import os, re
import glob
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
from nltools.data import Brain_Data, Design_Matrix
from nltools.stats import regress, zscore
from nltools.file_reader import onsets_to_dm
from nltools.external import glover_hrf
import time      # monitor running time
import psutil    # monitor RAM usage
import sys       # read the command line argument (job ID)

# ---------------------------------------------------------------------
#                            function
# ---------------------------------------------------------------------
def make_motion_covariates(mc, tr):
    '''Create motion covariates regressors from realignment parameters
    Args:
        mc: (pd.DataFrame) realignment parameters
        tr: (float) repetition time
    Returns:
        mcReg: (Design_Matrix) instance that contains all 24 motion covariates
    '''
    z_mc = zscore(mc)
    all_mc = pd.concat([z_mc, z_mc**2, z_mc.diff(), z_mc.diff()**2], axis=1)
    all_mc.fillna(value=0, inplace=True)
    mcReg = Design_Matrix(all_mc, sampling_freq=1/tr)
    mcReg.columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'trans_x2',
       'trans_y2', 'trans_z2', 'rot_x2', 'rot_y2', 'rot_z2', 'trans_x3', 'trans_y3',
       'trans_z3', 'rot_x3', 'rot_y3', 'rot_z3', 'trans_x4', 'trans_y4', 'trans_z4',
       'rot_x4', 'rot_y4', 'rot_z4']
    return mcReg

# ---------------------------------------------------------------------
#                 parameters (change per time running)
# ---------------------------------------------------------------------
dataDir = ''    # where the smoothed, preprocessed BOLD data are stored
covDir = ''    # fmriprep output directory, which contains the confounds files
tomEventsDir = ''    # events files for theory of mind ratings generated from behavioral data
socEventsDir = ''
outputDir = ''

process = psutil.Process()

# a list of subjects with useable data in ses-02
##### MORDIFY THE FOLLOWING LINE WITH BATCH JOB ARRANGEMENT AND RUNNING GOALS
jobID = int(sys.argv[1]) - 1
# subList = subList[jobID*8:(jobID+1)*8] if jobID < 11 else subList[jobID*8:]    # 12 jobs, 8 or less subjects per job
# subList = ['sub-0023']    # test
subList = pd.read_csv('')['subjects']  # rerun some subjects
subList = [subList[jobID]]
#####
runList = ['1', '2', '3', '4']
# runList = ['2', '3']     # test

tr = 0.46    # TR = 0.46s
nDummy = 6   # number of dummy volumes at the start of each run
start_time = time.time()
i = 1

##### BAD RUNS
badruns = pd.read_csv(os.path.join(tomEventsDir, 'problematic_runs_narratives.csv'))

# ---------------------------------------------------------------------
#                            main loop
# ---------------------------------------------------------------------
for sub in subList:
    print('***********************************', flush=True)
    print(f'Subject #{i} in this job', flush=True)
    allBold = Brain_Data()    # a Brain_Data instance containging the BOLD data of all runs of a single subject 
    allDm = Design_Matrix(sampling_freq = 1/tr)    # a multi-run design matrix
    stats = []
    data = []
    nodataCount = 0    # if all events files were not available, don't do regressions
    
    for run in runList:
        # skip bad runs
        if sub in list(badruns['sub']):
            runInorNot = ('run-'+run == badruns.loc[badruns['sub']==sub, 'run'])
            if runInorNot.any():
                nodataCount += 1
                continue

        # find run length (unit: TR)
        boldFile = os.path.join(dataDir, f'{sub}_ses-02_task-narratives_acq-mb8_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold_smoothed-fwhm6mm.nii.gz')
        if not os.path.isfile(boldFile):
            print(f'There is no bold file for {sub} run-{run}!')
            nodataCount += 1
            continue
        numberTr = nib.load(boldFile).shape[-1]    # run length (dummy volumes already excluded before saving this smoothed file)

        # check whether events files exist
        tomFile = os.path.join(tomEventsDir, f'{sub}_run0{run}.csv')
        socFile = os.path.join(socEventsDir, f'{sub}_run0{run}.csv')
        if not os.path.isfile(tomFile) or not os.path.isfile(socFile):
            print(f'There is no events file for {sub} run-0{run}!')
            nodataCount += 1
            continue
        
        # load the two events files as design matrices, and remove duplicate columns (rating and spikes)
        tomEvents = Design_Matrix(pd.read_csv(tomFile), sampling_freq = 1/tr)
        socEvents = Design_Matrix(pd.read_csv(socFile), sampling_freq = 1/tr)
        socEvents = socEvents.drop(columns=['rating'])   # remove rating
        spike_columns = [col for col in tomEvents.columns if re.match(r"spike\d+_run\d+", col)]    # spike column names
        sums = tomEvents[spike_columns].sum(axis=1)    # the sum of the spike columns
        columnsToDrop = []    # if a spike column in socEvents is already in tomEvents, drop it
        for i in range(1, socEvents.shape[1]):
            idx = np.where(socEvents.iloc[:, i] == 1)[0][0]
            if sums[idx] == 1:
                columnsToDrop.append(socEvents.columns[i])
        socEvents_updated = socEvents.drop(columns=columnsToDrop)
        for i in range(1, socEvents_updated.shape[1]):    # if a spike column in socEvents is not in tomEvents, rename it to avoid duplicate columns
            socEvents_updated = socEvents_updated.rename(columns={socEvents_updated.columns[i]: f"{socEvents_updated.columns[i]}_SInt"})
        
        # concatenate the two design matrices and convolve the ratings
        dm = Design_Matrix(pd.concat([tomEvents, socEvents_updated], axis=1), sampling_freq = 1/tr)
        dm = dm.reset_index(drop=True)
        dmCon = dm.convolve(columns=['ToM_audio', 'SInt_audio', 'rating']) if run in ['1', '2'] else dm.convolve(columns=['ToM_text', 'SInt_text', 'rating'])
        
        # nuisance regressors
        # motion covariates
        covFile = os.path.join(covDir, sub, 'ses-02', 'func', f'{sub}_ses-02_task-narratives_acq-mb8_run-{run}_desc-confounds_timeseries.tsv')
        cov = pd.read_csv(covFile, sep='\t')
        cov = cov.loc[nDummy:]
        cov = cov.reset_index(drop=True)
        mc = cov[['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z']]
        mcReg = make_motion_covariates(mc, tr=tr)
        # csf signals
        csf = cov[['csf']]
        csf = Design_Matrix(csf, sampling_freq=1/tr)
        covs = Design_Matrix(pd.concat([mcReg, csf], axis=1), sampling_freq=1/tr)
        
        data = Brain_Data(boldFile)
        print(f'{sub} run-0{run}: BOLD data loaded', data.shape())
        spikes = data.find_spikes(global_spike_cutoff=3, diff_spike_cutoff=3)
        spikes = Design_Matrix(spikes.iloc[:,1:], sampling_freq=1/tr)
        # rename the spikes so that they won't be in the same regressors in the all-run dm
        for col in spikes.columns:
            spikes = spikes.rename(columns={col:f'{run}_{col}'})

        # runwise grand mean scaling
        grand_mean = np.mean(data.mean().data)
        data.data = 100 * data.data / grand_mean

        allBold = allBold.append(data)
        print(f"Current data size: {allBold.shape()}", flush=True)

        # high-pass filter
        dmCon = dmCon.add_dct_basis(duration=128)
        for col in dmCon.columns:
            if col[:3] == 'cos':
                 dmCon = dmCon.rename(columns={col:f'{run}_{col}'})

        # design matrix containing all event regressors and nuisance regressors
        dmCon = Design_Matrix(pd.concat([dmCon, covs, spikes], axis=1), sampling_freq=1/tr)
        # append the design matrix to all run matrix, create separate columns for the covariates
        allDm = allDm.append(dmCon, axis=0, unique_cols=covs.columns)
    
    print(f'{sub}: All design matrices concatenated')

    # regression
    if nodataCount == 4:
        print(f'No events or BOLD data for {sub}!')
        continue
    allBold.X = allDm
    stats = allBold.regress()

    # save beta maps
    indToM_audio = np.where(allDm.columns=='ToM_audio_c0')
    indToM_text = np.where(allDm.columns=='ToM_text_c0')
    indSocInt_audio = np.where(allDm.columns=='SInt_audio_c0')
    indSocInt_text = np.where(allDm.columns=='SInt_text_c0')
    indRating = np.where(allDm.columns=='rating_c0')
    stats['beta'][indToM_audio].write(os.path.join(outputDir, f'{sub}_ToM_audio_beta.nii.gz'))
    stats['beta'][indToM_text].write(os.path.join(outputDir, f'{sub}_ToM_text_beta.nii.gz'))
    stats['beta'][indSocInt_audio].write(os.path.join(outputDir, f'{sub}_SInt_audio_beta.nii.gz'))
    stats['beta'][indSocInt_text].write(os.path.join(outputDir, f'{sub}_SInt_text_beta.nii.gz'))
    stats['beta'][indRating].write(os.path.join(outputDir, f'{sub}_rating_beta.nii.gz'))
    print(f'{sub}: Writing done!')
    print(f"Current RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds, or {elapsed_time/60:.2f} minutes")
    
    i += 1