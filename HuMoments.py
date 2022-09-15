# todos:
# make this script to work in parallel, so that we do not have to wait 1h every time we use it. However, the code itself uses ~50 CPUs.

import numpy as np
from astropy.io import fits
import cv2
import h5py
from tqdm import tqdm


new_resolution = int(320)

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


##### load dataset
params = np.loadtxt('/home2/weiguang/Project-300-Clusters/ML/DS_reselected_all_halos.txt')
colums = 'rID[0] Hid[1]  R200: DS[2], eta[3], detla[4], fm[5], fm2[6];   R500: DS[7], eta[8], detla[9], fm[10], fm2[11]'

##### set the important parameters for R=R_{200}
eta = params[:,3]
delta = params[:,4]
fs1 = params[:,5] ## the sum of all substructure mass fraction
fs2 = params[:,6] ## only for the most massive substructure
#### for R = R_500
eta500 = params[:,8]
delta500 = params[:,9]
fs500 = params[:,10]

def chi(delta,fs):
    return np.sqrt(2/((delta/0.1)**2+(fs/0.1)**2))
chi_parameter = chi(delta,fs1)


selecth = np.load('/home2/weiguang/Project-300-Clusters/ML/Reselected_all_halos.npy')

hus = []
M_200 = []
M_3D = []
M_CIL = []
region_list = []
hids_list = []

for lp in tqdm(range(1,325)):
    #print('reading data region =' , lp)
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
            #print(lp,hid,RA)
            img = read_sz(lp,hid,RA)
            
            
            epsilon=1e-8
            new_img = (np.log((img+epsilon)/epsilon)/np.log(1/epsilon))

            moments = cv2.moments(new_img) 
            # Calculate Hu Moments 
            hu = cv2.HuMoments(moments)

            hus.append(-np.sign(hu)*np.log10(np.abs(hu)))
            
            # these resize function should be modified by a rebining function for obvious reasons. We dont need the
            # mean values but rather, the total sum of the emmision over the beam.
#             resized_xr = cv2.resize(img_xr+1e-20,(new_resolution,new_resolution))
#             resized_sz = cv2.resize(img_sz+1e-20,(new_resolution,new_resolution))
#             resized_dm = cv2.resize(img_dm+1e-20,(new_resolution,new_resolution))
#             resized_star = cv2.resize(img_star+1e-20,(new_resolution,new_resolution))
#             resized_mass = cv2.resize(img_mass+1e-20,(new_resolution,new_resolution))

           
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



hus = np.array(hus)
hus_list = hus.copy()

M_200 = np.array(M_200)
M_3D = np.array(M_3D)
M_CIL = np.array(M_CIL)
region_list = np.array(region_list)
hids_list = np.array(hids_list)

#hus = hus.reshape((2580,7,29))

hus = np.zeros(((2580,7,29)))
counter = 0
for i in range(2580):
    for k in range(29):
        hus[i,:,k] = hus_list[counter,:].flatten()
        counter+=1
        

print('creating data set that consists of the following keys...')



h5_path = path + "h5files/"
df = h5py.File(h5_path+'SZhus.h5', 'w')
df.create_dataset('hus', data = hus)
df.create_dataset('M_200',data = M_200)
df.create_dataset('chi_3D',data = chi_parameter)
df.create_dataset('M_3Dsphere',data = M_3D)
df.create_dataset('M_CIL',data = M_CIL)
df.create_dataset('region',data = region_list)
df.create_dataset('hid',data=hids_list)


df.close()