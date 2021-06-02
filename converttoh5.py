# todos:
# make this script to work in parallel, so that we do not have to wait 1h every time we use it. However, the code itself uses ~50 CPUs.

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from utils import plot_cluster
import cv2
import h5py

new_resolution = int(128)

path = "/data7/users/deandres/newML2/"

RAs = np.arange(0,29)


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

images_xr = []
images_sz = []
images_dm = []
images_star = []
M_200 = []
hids_list = []
selecth = np.load('/home2/weiguang/Project-300-Clusters/ML/Reselected_all_halos.npy')

for lp in range(1,325):
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
            
            resized_xr = cv2.resize(img_xr+1e-20,(new_resolution,new_resolution))
            resized_sz = cv2.resize(img_sz+1e-20,(new_resolution,new_resolution))
            resized_dm = cv2.resize(img_dm+1e-20,(new_resolution,new_resolution))
            resized_star = cv2.resize(img_star+1e-20,(new_resolution,new_resolution))
            
            
            images_xr.append(resized_xr)
            images_sz.append(resized_sz)
            images_dm.append(resized_dm)
            images_star.append(resized_star)
            #end RAs loop
        M_2 = get_M2(lp,hid,RA)   
        M_200.append(M_2) 
        hids_list.append(hid)
        #end Hids loop        
    #end region loop



images_xr = np.array(images_xr)
images_sz = np.array(images_sz)
images_dm = np.array(images_dm)
images_star = np.array(images_star)
M_200 = np.array(M_200)
hids_list = np.array(hids_list)

images_xr = images_xr.reshape((2580,29,new_resolution,new_resolution))
images_sz = images_sz.reshape((2580,29,new_resolution,new_resolution))
images_dm = images_dm.reshape((2580,29,new_resolution,new_resolution))
images_star = images_star.reshape((2580,29,new_resolution,new_resolution))

print('creating data set that consists of the following keys...')
print('shape x-ray : ',images_xr.shape)
print('shape sz : ',images_sz.shape)
print('shape dm : ', images_dm.shape)
print('shape M_200: ',M_200.shape)

h5_path = path + "h5files/"
df = h5py.File(h5_path+'128.h5', 'w')
df.create_dataset('Xray', data = images_xr)
df.create_dataset('SZ',data = images_sz)
df.create_dataset('DM',data = images_dm)
df.create_dataset('star',data = images_star)
df.create_dataset('M_200',data = M_200)
df.create_dataset('hid',data = hids_list)
df.close()
    
    