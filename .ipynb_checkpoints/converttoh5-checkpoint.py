# todos:
# make this script to work in parallel, so that we do not have to wait 1h every time we use it. However, the code itself uses ~50 CPUs.

import numpy as np
from astropy.io import fits
#import cv2
import h5py
from tqdm import tqdm
from utils import *
new_resolution = int(320)

path = "/data7/users/deandres/newML2/"

RAs = np.arange(0,29)




##### load dataset
params = np.loadtxt('/home2/weiguang/Project-300-Clusters/ML/DS_reselected_all_halos.txt')
colums = 'rID[0] Hid[1]  R200: DS[2], eta[3], detla[4], fm[5], fm2[6];   R500: DS[7], eta[8], detla[9], fm[10], fm2[11]'

##### set the important parameters for R=R_{200}

rID_chi = np.int64(params[:,0])
hid_chi = np.int64(params[:,1])

eta = params[:,3]
delta = params[:,4]
fs1 = params[:,5] ## the sum of all substructure mass fraction
fs2 = params[:,6] ## only for the most massive substructure
#### for R = R_500
eta500 = params[:,8]
delta500 = params[:,9]
fs500 = params[:,10]


chi_parameter = chi(delta,fs1)
h = 0.677


images_xr = []
images_sz = []
images_dm = []
images_star = []
images_mass = []
images_gas = []
chi_list = []
M_200 = []
R_200 = []
M_3D = []
M_CIL = []
region_list = []
hids_list = []
selecth = np.load('/data7/users/deandres/ML4/halosML4.npy')

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
        
        M_2 = get_M2(lp,hid,0)   
        R_2 = get_R2(lp,hid,0) 
        M_C = get_MCIL(lp,hid,0)
        M_3 = get_M3D(lp,hid,0)
        M_200.append(M_2) 
        R_200.append(R_2)
        M_3D.append(M_3)
        M_CIL.append(M_C)
        region_list.append(lp)
        hids_list.append(hid)
        
        chi_list.append(chi_parameter[(rID_chi==lp)&(hid_chi==hid)].flatten())
        
        
        for RA in RAs:
            #print(lp,hid,RA)
            img_xr = read_xlum(lp,hid,RA)
            img_sz = read_sz(lp,hid,RA)
            img_dm = read_dm(lp,hid,RA)
            img_star = read_star(lp,hid,RA)
            img_mass = read_total_mass(lp,hid,RA)
            img_gas = read_gas(lp,hid,RA)
            
            # normalise mass maps
            pixel_area = (2*R_2/h/new_resolution)**2 # in kpc**2
            img_dm = img_dm/pixel_area # 1e10 * M_sun / h / kpc**2
            img_star = img_star/pixel_area
            img_mass = img_mass/pixel_area
            img_gas = img_gas/pixel_area
            
            resized_xr = rebin(img_xr,(new_resolution,new_resolution))
            resized_sz = rebin(img_sz,(new_resolution,new_resolution))
            resized_dm = rebin(img_dm,(new_resolution,new_resolution))
            resized_star = rebin(img_star,(new_resolution,new_resolution))
            resized_mass = rebin(img_mass,(new_resolution,new_resolution))
            resized_gas = rebin(img_gas,(new_resolution,new_resolution))

            
            images_xr.append(resized_xr)
            images_sz.append(resized_sz)
            images_dm.append(resized_dm)
            images_star.append(resized_star)
            images_mass.append(resized_mass)
            images_gas.append(resized_gas)
    
            #end RAs loop

        #end Hids loop        
    #end region loop

images_xr = np.array(images_xr)
images_sz = np.array(images_sz)
images_dm = np.array(images_dm)
images_star = np.array(images_star)
images_mass = np.array(images_mass)
images_gas = np.array(images_gas)

M_200 = np.array(M_200)
R_200 =  np.array(R_200)
M_3D = np.array(M_3D)
M_CIL = np.array(M_CIL)
region_list = np.array(region_list)
hids_list = np.array(hids_list)
chi_list = np.array(chi_list)
n_clusters = len(M_200)

images_xr = images_xr.reshape((n_clusters,29,new_resolution,new_resolution))
images_sz = images_sz.reshape((n_clusters,29,new_resolution,new_resolution))
images_dm = images_dm.reshape((n_clusters,29,new_resolution,new_resolution))
images_star = images_star.reshape((n_clusters,29,new_resolution,new_resolution))
images_mass = images_mass.reshape((n_clusters,29,new_resolution,new_resolution))
images_gas = images_gas.reshape((n_clusters,29,new_resolution,new_resolution))

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
df = h5py.File(h5_path+'dataset_320_19Oct2022.h5', 'w')
df.create_dataset('Xray', data = images_xr)
df.create_dataset('SZ',data = images_sz)
df.create_dataset('DM',data = images_dm)
df.create_dataset('star',data = images_star)
df.create_dataset('total_mass',data = images_mass)
df.create_dataset('gas',data = images_gas)
df.create_dataset('M_200',data = M_200)
df.create_dataset('R_200',data = R_200)
df.create_dataset('chi_3D',data = chi_list )
df.create_dataset('region',data = region_list)
df.create_dataset('hid',data=hids_list)


df.close()