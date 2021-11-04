
# todos:
# make this script to work in parallel, so that we do not have to wait 1h every time we use it. However, the code itself uses ~50 CPUs.

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from utils import plot_cluster
import cv2
import h5py
from tqdm import tqdm

new_resolution = int(128)

path = "/data7/users/deandres/newML2/"

RAs = np.arange(0,29)


def read_total_mass(lp,hid,RA):
    region = 'total-mass/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Mtotal-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data


def read_xr(lp,hid,RA):
    region = 'X-ray/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Athena-wfi-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data

def read_sz(lp,hid,RA):
    region = 'SZ/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-TT-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data

def read_dm(lp,hid,RA):
    region = 'DM/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-DM-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data

def read_star(lp,hid,RA):
    region = 'star/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Mstar-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    return data

def get_M2(lp,hid,RA):
    region = 'DM/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-DM-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    hdul = fits.open(path+region+file)
    M = np.float(hdul[0].header[-2][12:18])
    return M

def get_M3D(lp,hid,RA):
    region = 'total-mass/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Mtotal-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    hdul = fits.open(path+region+file)
    header = hdul[0].header
    return float(header[-1][-8:])

def get_MCIL(lp,hid,RA):
    region = 'total-mass/NewMDCLUSTER_{}/'.format(str(lp).zfill(4))
    s = str(hid)[:3]
    file = 'snap_{}-Mtotal-cl-{}-ra-{}.fits'.format(s,hid,RA)
    #print(RA)
    data = fits.getdata(path+region+file)
    hdul = fits.open(path+region+file)
    header = hdul[0].header
    return float(header[-2][-8:])

images_xr = []
images_sz = []
images_dm = []
images_star = []
images_mass = []
M_200 = []
M_3D = []
M_CIL = []
region_list = []
hids_list = []
selecth = np.load('/home2/weiguang/Project-300-Clusters/ML/Reselected_all_halos.npy')

for lp in tqdm(range(1,325)):
    print('reading data region =' , lp)
    idr = np.where(np.int32(selecth[:,2]+0.1)==lp)[0]
    if len(idr)<1:
        raise ValueError('No regions find in selected halo',lp)

    Hids = np.int64(selecth[idr,0]+0.1)    #AHF halo IDs
    sn = np.array([np.int32(str(i)[:3]) for i in Hids])
    idshid = np.argsort(Hids)
    Hids = Hids[idshid]; sn=sn[idshid]; idr=idr[idshid]
    st = 0
    
    for hid in Hids:
        for RA in RAs:
            img_xr = read_xr(lp,hid,RA)
            img_sz = read_sz(lp,hid,RA)
            img_dm = read_dm(lp,hid,RA)
            img_star = read_star(lp,hid,RA)
            img_mass = read_total_mass(lp,hid,RA)
            
            resized_xr = cv2.resize(img_xr+1e-20,(new_resolution,new_resolution))
            resized_sz = cv2.resize(img_sz+1e-20,(new_resolution,new_resolution))
            resized_dm = cv2.resize(img_dm+1e-20,(new_resolution,new_resolution))
            resized_star = cv2.resize(img_star+1e-20,(new_resolution,new_resolution))
            resized_mass = cv2.resize(img_mass+1e-20,(new_resolution,new_resolution))
            
            
            images_xr.append(resized_xr)
            images_sz.append(resized_sz)
            images_dm.append(resized_dm)
            images_star.append(resized_star)
            images_mass.append(resized_mass)
            #end RAs loop
        M_2 = get_M2(lp,hid,RA)   
        M_C = get_MCIL(lp,hid,RA)
        M_3 = get_M3D(lp,hid,RA)
        M_200.append(M_2) 
        M_3D.append(M_3)
        M_CIL.append(M_C)
        region_list.append(lp)
        hids_list.append(hid)
        #end Hids loop        
    #end region loop



images_xr = np.array(images_xr)
images_sz = np.array(images_sz)
images_dm = np.array(images_dm)
images_star = np.array(images_star)
images_mass = np.array(images_mass)

M_200 = np.array(M_200)
M_3D = np.array(M_3D)
M_CIL = np.array(M_CIL)
region_list = np.array(region_list)
hids_list = np.array(hids_list)

images_xr = images_xr.reshape((2580,29,new_resolution,new_resolution))
images_sz = images_sz.reshape((2580,29,new_resolution,new_resolution))
images_dm = images_dm.reshape((2580,29,new_resolution,new_resolution))
images_star = images_star.reshape((2580,29,new_resolution,new_resolution))
images_mass = images_mass.reshape((2580,29,new_resolution,new_resolution))

print('creating data set that consists of the following keys...')
print('shape x-ray : ',images_xr.shape)
print('shape sz : ',images_sz.shape)
print('shape DM : ', images_dm.shape)
print('shape star : ', images_star.shape)
print('shape total mass : ', images_mass.shape)
print('shape M_200: ',M_200.shape)
print('shape region: ',region_list.shape)
print('shape hid: ',hids_list.shape)
print('integration factor ',(640/new_resolution)**2)

h5_path = path + "h5files/"
df = h5py.File(h5_path+'128_totalmass.h5', 'w')
df.create_dataset('Xray', data = images_xr)
df.create_dataset('SZ',data = images_sz)
df.create_dataset('DM',data = images_dm)
df.create_dataset('star',data = images_star)
df.create_dataset('total_mass',data = images_mass)

df.create_dataset('M_200',data = M_200)
df.create_dataset('M_3Dsphere',data = M_3D)
df.create_dataset('M_CIL',data = M_CIL)
df.create_dataset('region',data = region_list)
df.create_dataset('hid',data=hids_list)
df.create_dataset('Ifactor',data=(640/new_resolution)**2)