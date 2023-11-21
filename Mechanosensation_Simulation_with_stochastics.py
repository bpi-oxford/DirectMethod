# Goal of this Simulation is to quantify the possible differences in estimated traction forces
# according to different stiffness of the used Gel as well as different optical systems. 
# therefore we check the stiffnesses 0.1, 0.4, 0.7, 1, and 5 kPa. for Forces between 0.01  and 1 nN; 
# Literature states forces between 0.025 and 0.5 nN (https://onlinelibrary.wiley.com/doi/full/10.1111/boc.202000133)
# BPI-Lab says: 0.05 to 0.35 nN (https://www.sciencedirect.com/science/article/pii/S2211124719302530?via%3Dihub) 
# 0.05 nN steps --> 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35.
# FOV : 36.6 x 36.5 um  
# Cell Size: diameter = 10 um. 
# mesh_size = 20 ; 15 x 15 = 225 datapoints (ca. 300 beads for aTFM after cleaning)
# spacing_xy = 0.2 um  # confocal
# spacing_z = 0.36 um # confocal 
#
# spacing_xy = 0.02 um # TIRF-SIM
# spacing_z =  0.09 um # TIRF-SIM 
# 
# Pipepline: 
# - Generate required toml files. # done
# - Run force field simulation (prefferalbly dounut) # done
# - Plot generated displacment field # WIP
# - Filter Displacement field 
# - Plot filtered Displacement field (# delete all displacements under resolution limit)
# - Run Force reconstruction, from displacement 2D FTTC


#%%
#Imports 

#%matplotlib ipympl
import matplotlib.pyplot as plt 
import numpy as np 
import toml 
import os, sys 

#os.chdir(r'C:\Users\marce\Documents\Mechanosensation_Simulation')
os.chdir(r'/media/Data2/Marcel/Mechanosensation_Simulation')
import shutil
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from src.TFM.uFieldType import load_from_ufile
from src.TFM import tfmFunctions
from skimage.restoration import estimate_sigma
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
import napari 
from multiprocessing import Pool
import time
import tifffile 
import trackpy as tp
import scipy
from skimage import measure

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    

#%% 

# Run force field simulation / Creat displacement_field_data

def raw_displacements(name):
    i = name
    
    # Clear input
    if 'description.toml' in os.listdir():
        os.remove('description.toml')
    
    # Grab descriptor from toml file
    file_location = 'toml_files/' + i

    shutil.copyfile(file_location, i)

    os.rename(i, 'description.toml')
    output_path = dis_loc + '/displacement_' + i
    do_sim = 'python geninput.py sim ' + str(noise) + ' ' + output_path 
    blockPrint()
    os.system(do_sim)
    enablePrint()
    os.remove('description.toml')
    

        
## This doesnt work because of the description toml import    
# with Pool() as p:
#     tqdm(p.map(raw_displacements,os.listdir('toml_files')))


# %% 
# Define filter with gauss, random bg noise and beads
def gaus2d(x, y, mx, my, sx, sy):
        return np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

def generate_image(beads, n_im ,fwhm, pxl_size):
 
    # create empty synthetic image:
    _image = np.zeros((n_im,n_im))
    #print(np.shape(_image))
    _zeros = _image.copy()

    #print(_width_psf)
    std = fwhm/pxl_size 
    _x_image= np.linspace(0,n_im,n_im)
    _xx_image, _yy_image = np.meshgrid(_x_image,_x_image)

    # plt.imshow(_psf*max_intensity)
    # plt.colorbar()
    # plt.show()

    # plt.plot(_psf)
    # plt.show()

    for b in beads:
        _x_bead = b[0] * n_im
        _y_bead = b[1] * n_im
        #print(_x_bead,_y_bead)
        _psf = gaus2d(_xx_image,_yy_image,_x_bead,_y_bead,std,std)

        #filter threshhold 
        # plt.imshow(_psf)
        # plt.title('_psf')
        # plt.show()

        _image = _image + _psf#*max_intensity
        
    # Post enditing 
    # convert values to photon counts    
    _image = np.round(_image * max_intensity /np.max(_image),0)
    _image = _image + 5
    noise_mask = np.random.poisson(_image)
    _image = _image + noise_mask
    #_image = (_image < 1) *_image
    #plt.imshow(_image)
    return _image

def synthetic_image_pairs(beads, dx, dy, resolution,pxl_size):
    
    # scale normed beads 
    _x_bead =  beads[:,0] * n_points_xy
    _y_bead =  beads[:,1] * n_points_xy
    __xx,__yy = np.meshgrid(np.linspace(0,n_points_xy, n_points_xy),np.linspace(0,n_points_xy, n_points_xy))

    _x_loc = _x_bead
    _y_loc = _y_bead

    # First interpolation determining displacemnet values in x ynd y for each bead 
    _ux_beads = griddata((__xx.ravel(),__yy.ravel()), dx.ravel(), (_x_bead,_y_bead), method = 'cubic')
    _uy_beads = griddata((__xx.ravel(),__yy.ravel()), dy.ravel(), (_x_bead,_y_bead), method = 'cubic')

    # Generate synthetic Image and use trackpy to retrace Traction field.
    # plt.scatter(_ux_beads* pixel_size*(n_points_xy)/1000,_uy_beads* pixel_size*(n_points_xy)/1000)
    # plt.show()
    #print(np.nan_to_num(_ux_beads))

    # Apply displacement field on bead position all values normed between 0 and 1, according to pixel number.
    # Displacement data is given in nm therefore it needs to multiplied by 1000.
    _uu_beads = beads.copy()
    _uu_beads[:,0] = _uu_beads[:,0] + np.nan_to_num(_ux_beads) /(pixel_size*n_points_xy*1000)
    _uu_beads[:,1] = _uu_beads[:,1] + np.nan_to_num(_uy_beads) /(pixel_size*n_points_xy*1000)


    # plt.scatter(_uu_beads[:,0],_uu_beads[:,1], c = 'blue')
    # plt.scatter(beads[:,0],beads[:,1], c= 'red')
    # plt.show()
    # print(_uu_beads)
    # generate synthetic images
    _image_1 = generate_image(beads, n_points_xy, resolution, pxl_size) 
    _image_2 = generate_image(_uu_beads, n_points_xy, resolution, pxl_size)


    return _image_1, _image_2
#%%

def pollute_displacement(name):

    # Load Displacement Field
    path = dis_loc + '/' + name
    load_array = np.load(path)
    array = dict(load_array)
    dx = array['ux'][:,:,0]
    dy = array['uy'][:,:,0]

    # Generate Beads
    np.random.seed() # needed so that the daughter processes do not use the same random variable
    beads = np.random.rand(n_beads,2)
    save_beads = 'beads/beads_' + str(name)
    
    # Save Beads
    #np.save(save_beads, beads)

    #Generate Synthetic Images and save them
    im1,im2 = synthetic_image_pairs(beads, dx, dy , microscope_resolution, pixel_size)
    file_name= os.getcwd() + '/' + tmp_image_loc + '/' + name[:-4] + '.tif'
    tifffile.imwrite(file_name, np.array((im1,im2)))

# for i in tqdm(os.listdir('displacement_fields')):
#     pollute_displacement(i)


    
#%%
#napari.view_image(np.array((im1,im2)))
#%%
# Save test_image_pair
# file_name= os.getcwd() +'/'+ image_loc + '/' +  os.listdir('displacement_fields')[0][:-4] + '.tif'
# tifffile.imwrite(file_name, np.array((im1,im2)))
# print(file_name)


#%%
beads = np.random.rand(n_beads,2)

#%% 

# Sainity check for filter method
test_imgen = False
if test_imgen == True:
    
    # Load displacment file
    test_path = 'displacement_fields/displacement_1_nN_on_1_kPa.npz'# + '#os.listdir('displacement_fields')[-1]
    file = np.load(test_path)
    print(test_path)
    # plt.scatter(beads[:,0],beads[:,1])
    # plt.show()
    
    # Load Synthetic image to sainity check.
    test_path = 'synthetic_images/displacement_1_nN_on_1_kPa.tif'# + os.listdir('displacement_fields')[-1] + '.tif'
    test_image = tifffile.imread(test_path)

    test_x = dict(file)['ux'][:,:,0]
    test_y = dict(file)['uy'][:,:,0]
    
    # plt.imshow(np.sqrt(test_x**2+test_y**2))
    # plt.show()
    
    im1,im2 = synthetic_image_pairs(beads, test_x, test_y, microscope_resolution, pixel_size)
    plt.imshow(test_image[0])
    plt.show()
    
    plt.imshow(test_image[1])
    plt.show()
    
    plt.imshow(test_image[0]-test_image[1])
    plt.show()
    
    r = int(microscope_resolution/pixel_size) 
    
    d_max = 10

    if r % 2 == 0: 
        r = r+ 1
    
    test_locate = tp.locate(test_image[0], r, separation = r)
    test_locate2 = tp.locate(test_image[1], r, separation = r)
    
    tp.annotate(test_locate, test_image[0])
    tp.annotate(test_locate2, test_image[1])
    
    test_set = tp.batch(test_image, r, separation = r)
    test_link = tp.link(test_set, d_max)
    
    test_particles = []
    x = []
    y = []
    ux = []
    uy = []
    
    for p in set(test_link['particle']):
        
        #Grab Particles
        two = test_link[test_link['particle']==p]
        
        #Check if Particle is in both images
        if len(two) > 1:
            t_0 = two[two['frame']==0].copy()
            t_1 = two[two['frame']==1].copy()
            #print(p)
            x.append(float(t_0['x']))
            y.append(float(t_0['y']))
            ux.append(float(t_0['x'])-float(t_1['x']))
            uy.append(float(t_0['y'])-float(t_1['y']))
    
    
    _xx,_yy = np.meshgrid(np.linspace(0,n_points_xy, n_points_xy),np.linspace(0,n_points_xy, n_points_xy))
    _uux = griddata((x,y),ux,(_xx,_yy) , method = 'cubic')
    _uuy = griddata((x,y),uy,(_xx,_yy) , method = 'cubic')
        
    _uux = np.nan_to_num(_uux, nan = 0)
    _uuy = np.nan_to_num(_uuy, nan = 0)
    
    plt.imshow(np.sqrt(_uux**2+_uuy**2))
    plt.show()
    tp.subpx_bias(test_set) # if we see dip in middle that is bad and there is a bias.
#%%
# Recover Displacement field from synthetic images using trackpy
# grab locations of identified bead and use displacement to generate displacement field.
#Iterate over particles: 
def generate_ux_uv(name):
    name_path = tmp_image_loc + '/' + name
    print(name_path)
    test_image = tifffile.imread(name_path)
    d = int(microscope_resolution/pixel_size) * 2 
    d_max = int(0.25/pixel_size) # Fix max linking distance at 250 nm.
    
    if d % 2 == 0: 
        d = d+ 1 

    test_set =  tp.batch(test_image, d, separation = d ) 
    test_link = tp.link(test_set, d_max)
    
    # Interate through particles
    test_particles = []
    x = []
    y = []
    ux = []
    uy = []
    
    for p in set(test_link['particle']):
        
        #Grab Particles
        two = test_link[test_link['particle']==p]
        
        #Check if Particle is in both images
        if len(two) > 1:
            t_0 = two[two['frame']==0].copy()
            t_1 = two[two['frame']==1].copy()
            #print(p)
            x.append(float(t_0['x']))
            y.append(float(t_0['y']))
            ux.append(float(t_0['x'])-float(t_1['x']))
            uy.append(float(t_0['y'])-float(t_1['y']))
    
    _xx,_yy = np.meshgrid(np.linspace(0,n_points_xy, n_points_xy),np.linspace(0,n_points_xy, n_points_xy))
    _uux = griddata((x,y),ux,(_xx,_yy) , method = 'cubic')
    _uuy = griddata((x,y),uy,(_xx,_yy) , method = 'cubic')
        
    _uux = np.nan_to_num(_uux, nan = 0)/pixel_size
    _uuy = np.nan_to_num(_uuy, nan = 0)/pixel_size
    
    # save as npz in required format.
    npz_blueprint = np.load(str(dis_loc + '/' + os.listdir(dis_loc)[0]))
    #print(dict(npz_blueprint)['ux'])
    
    npz = dict(npz_blueprint)
    # print(np.shape(npz['ux']))
    for i in range(n_points_z):
        npz['ux'][:,:,i] = _uux
        npz['uy'][:,:,i] = _uuy
        npz['uz'][:,:,i] = npz['uz'][:,:,i][:len(npz['uz'])]
        npz['uAbs'][:,:,i] = np.sqrt(_uux**2+_uuy**2)
    save_path = tmp_filter_loc + '/filtered_' + name[:-4]
    np.savez(save_path, **npz)
    # print(save_path)
    # for i in npz.keys():
    #     print(np.shape(npz[i]))
    return


    

#%%
# run FTTC3d in skript: 

# choose solver
runtfm = getattr(tfmFunctions, 'FTTC') # keep it in 2d, keep it simple 

def tfm(path):

    DVCPath = path
    
    blockPrint()
    uCurr = load_from_ufile(DVCPath)
    
    grid, UVec, UzVec, QVec = runtfm(uCurr)
    enablePrint()
    
    us, vs, ws = UVec
    qx, qy, qz = QVec
    qAbs = np.sqrt(qx * qx + qy * qy)
    traction_array = {  'us': us, 'vs': vs, 'ws': ws,
                        'qx': qx, 'qy': qy, 'qz': qz, 
                        'qAbs': qAbs}
    np.savez(tmp_trac_loc+ '/tractions_' + path[path.index('/')+1:-4] + '.npz', **traction_array)

#%%
def svd_metric(im1,im2): 
    
    # metric returns normalised distance between two images
    # lower is better 
    U1, s1, Vh1 = scipy.linalg.svd(im1)
    U2, s2, Vh2 = scipy.linalg.svd(im2)
    metric = np.sqrt(np.sum((s1-s2)**2))/(len(s1)*len(s1))
    
    return metric
#%%
def accuracy_image(_im,_gt): 
    def _count_objects(_gray):
        _binary_image = _gray >(.8*np.max(_gray))# for gt 80 % returns area of ca. 1.05 um**2
        _contours = np.array(measure.find_contours(_binary_image))
        _contours = np.array([c for c in _contours if len(c)> 50])

        return len(_contours)

    if _count_objects(_gt) == _count_objects(_im):
        return 1 

    else:
        return 0

def precision_image(_im,_gt):

    # peak traction within large objects within 20% of max ground truth
    #_mask = _im > 0.8*np.mean(_im)
    #_im = _im * _mask

    if np.max(_gt) * .8 <= np.max(_im) < np.max(_gt) * 1.2:
        _precision = 1

    else: 
        _precision = 0

    return _precision

# print(accuracy_image(image,gt))
# print(precision_image(image,gt))
# print('sainity check')
# print(precision_image(gt,gt))
# print(accuracy_image(gt,gt))

def traction(F,E):
    array = np.zeros(np.shape(xx))
    print(array)
    k = 0
    for e in range(len(E)): 
        n = 0
        for  f in range(len(F)): 
            name = 'tractions/tractions_displacement_' + str(F[f]) + '_nN_on_' + str(E[e]) + '_kPa.npz'
            print(name)
            file = dict(np.load(name))
            #print(file.keys())
            array[e][f] = np.max(file['qAbs'])
            print(f,e)
            #plt.imshow(file['qAbs'])
            #plt.show()
            n = n+1
        k = k+1
    return array

def filtered_traction(F,E):
    array = np.empty(np.shape(xx))
    #print(array)
    k = 0
    for e in range(len(E)): 
        n = 0
        for  f in range(len(F)): 
            name = 'tractions/tractions_filtered_displacement_' + str(F[f]) + '_nN_on_' + str(E[e]) + '_kPa.npz'
            file = dict(np.load(name))
            array[f][e] = np.max(file['qAbs'])
            #print(k,n,'\n',f,e)
            #plt.imshow(file['qAbs'])
            #plt.show()
            n = n+1
        k = k+1
    return array

def main():
    start_time = time.time()

    # create output directories
    tom_loc = r'toml_files'
    os.makedirs(tom_loc, exist_ok=True)

    dis_loc = r'displacement_fields'
    os.makedirs(dis_loc, exist_ok=True)
        
    filter_loc = r'filtered_displacement_fields'
    os.makedirs(filter_loc,exist_ok=True)

    image_loc = r'synthetic_images'
    os.makedirs(image_loc, exist_ok=True)

    plot_loc = r'plotting'
    os.makedirs(plot_loc, exist_ok=True)
        
    trac_loc = r'tractions'
    os.makedirs(trac_loc, exist_ok=True)

    bead_loc = r'beads'
    os.makedirs(bead_loc, exist_ok=True)
        
    #Define parameter space
    microscope_resolution = 0.18 # in um Confocal
    #microscope_resolution = 0.07 # in um TIRF-SIM
 
    #young_list = np.logspace(-1,2,num=100, endpoint=1)# in kPa 
    young_list = [i*0.2 for i in range(1,21)]
    force_list = np.linspace(0.001,1,20) # in nN 
 
    #young_list = [0.4, 1,10]
    # force_list = [1,5]
    #force_list = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1][::2]
    #force_list = [1,10,100]
    #force_list = np.logspace(-2,3,100)
    poisson_ratio = .49 # possoin ratio
 
    scale = 0.03 # 70 nm per  pxl for TIRF-SIM
 
    # Field of view is 3.78 x 3.78 um
    # indenters have a diameter of 1 um and are 0.5 um apart. 
    # indenter have a diameter of 500nm and are 0.25 um apart. 
    # --> this should show the case where resolution should resolve everything as desired, but reconstruction might be effected
    sigma = 1 # beads per µm^2
    n = 100
 
    corner_length = 7.92 # µm
    n_beads = round(sigma * corner_length**2)
    max_intensity = 4096
    spacing_z = 0.36 
    n_points_z = 4 # apparently this skript requires to build a third dimension.
    # This is carried through till FTTC evaluation
    pixel_size = 0.03
    n_points_xy = round(corner_length / pixel_size)
    bg  = 0.17 #Background in pixeln
    noise = 0
 
    # Plot parameter space
    xx_p,yy_p = np.meshgrid(young_list, force_list)
    plt.figure(dpi = 150)
    plt.scatter(xx_p,yy_p)
    plt.xlabel('E[kPa]')
    plt.ylabel('F[nN]')
    plt.title('Parameter space')
    plt.show()
  
 
    toml_file = {'dataset': {'name': 'Test'},
     'substrate': {'E': 50000.0, 'nu': poisson_ratio},
     'image': {'spacing_xy': scale, 'spacing_z': 0.02},
     'simulation': { 'n_points_xy': n_points_xy, 'n_points_z': n_points_z, 'NBeads': n_beads},
     'adheasion': {'indenter': {'type': 'dipole',
       'F': 10000000.0,
       #'Fz':10000000.0,
       'd': 2,
       'phi': 45,
       'a': 1,
       'pos': [0, 0]}}}
 
    xx, yy = np.meshgrid(np.arange(n_points_xy),np.arange(n_points_xy))
 
    # Create toml files: 
    for E in young_list: 
        for F in force_list: 
            toml_file['dataset']['name'] = str(F) + '_nN_on_' + str(E) + '_kPa'
            toml_file['substrate']['E'] = E
            #toml_file['image']['micronPerPixel'] = 0.7 # TIRF-SIM aTFM Record
            toml_file['adheasion']['indenter']['F'] = F 
            #toml_file['adheasion']['indenter']['Fz'] = F 
            toml_file['adheasion']['indenter']['F_unit'] = 'nN'
            # save toml_file
            output_toml_name = 'toml_files/' + toml_file['dataset']['name']
            toml_string = toml.dumps(toml_file)
            with open(output_toml_name, "w") as output_toml:
                toml.dump(toml_file, output_toml)
                
    #%%
    for i in tqdm(os.listdir(tom_loc)):
        raw_displacements(i)
    #%%
    n = 100    
    for i in range(1,n+1):
    
        tmp_filter_loc = filter_loc + '/' + str(i) + '_run'
        if not os.path.exists(tmp_filter_loc):
            os.makedirs(tmp_filter_loc)
            
        tmp_image_loc = image_loc + '/' + str(i) + '_run'
        if not os.path.exists(tmp_image_loc):
            os.makedirs(tmp_image_loc)
    
        tmp_trac_loc = trac_loc + '/' + str(i) + '_run'
        if not os.path.exists(tmp_trac_loc):
            os.makedirs(tmp_trac_loc)
      
        # Create filter set
        print('\nGenerating experimental displacement field data run: ', i)
    
        p = Pool()
        for _ in tqdm(p.map(pollute_displacement, os.listdir(dis_loc)), total=len(os.listdir(dis_loc))):
            pass
    
        print('done')    
    
        #Run Displacment_field Generation
        #parralelization is not allowed deamon children, check when you have time 
        # p = Pool()
        # for _ in tqdm(p.map(generate_ux_uv, os.listdir(image_loc)), total=len(os.listdir(image_loc))):
        #     pass
        for i in tqdm(os.listdir(tmp_image_loc)):
            generate_ux_uv(i)  
            
            
        solve = False
        if solve == True:   
            # solve raw-dataset
            print('solving for raw files')
       
            p = Pool()
            for _ in tqdm(p.map(tfm, [str(dis_loc+'/'+ i ) for i in os.listdir(dis_loc)]), total=len(os.listdir(dis_loc))):
                pass
       
            print('raw files - done \n\nsolving for filtered files')
       
       
            # solve filtered-dataset
            p = Pool()
            for _ in tqdm(p.map(tfm, [str(tmp_filter_loc+'/' + i ) for i in os.listdir(tmp_filter_loc)]), total=len(os.listdir(tmp_filter_loc))):
                pass
            print('traction calculation - done')  
     
        
    print("--- %s seconds ---" % (time.time() - start_time))
        
    #%%
    # plot displacement profiles 
    noise_list = []
    noise_list_raw = []
    mean_traction_list = []
    peak_traction_filtered = []
    peak_tractions = []
    mean_raw = []
    
    svd_list = []
    accuracy_list = []
    precision_list = []
    
    n = 100
    for i in tqdm(range(1,n+1)):
        tmp_dis_loc = dis_loc + str(i) +'_run'
        tmp_filter_loc = filter_loc +'/' + str(i) +'_run'
        
        tmp_noise_list = []
        tmp_mean_traction_list = []
        tmp_peak_traction_filtered = []
        tmp_mean = []
        tmp_svd_list = []
        tmp_peak_tractions = []
        tmp_accuracy_list = []
        tmp_precision_list = []
        
        for f in force_list:
            for e in young_list:
                path = dis_loc + '/displacement_' + str(f) + '_nN_on_' + str(e) +'_kPa.npz'
                path_filtered = tmp_filter_loc + '/filtered_displacement_' + str(f) + '_nN_on_' + str(e) +'_kPa.npz'
                T = np.load(path)['uAbs'][:,:,0]
                
                #T_filt = os.listdir(fiter_loc)
                
                #T_mask = (T < (0.04))
                #T = T_mask * T
                T_filt = np.load(path_filtered)['uAbs'][:,:,0]
                
                # determine and append noise:
                # using skimage estimator:    
                tmp_noise_list.append(estimate_sigma(T_filt))
                # using mean of a low impact window at upper right 80-90 % corner length:
                T_crop = T_filt[ int(0.2*len(T_filt)) : int(0.8*len(T_filt))
                                ,int(0.2*len(T_filt)) : int(0.8*len(T_filt))]
                T_crop_raw = T[ int(0.2*len(T)) : int(0.8*len(T))
                                ,int(0.2*len(T)) : int(0.8*len(T))]
                
                #print('bg_traction: ',noise[-1], ' Pa')
    
                tmp_mean_traction_list.append(np.mean(T_filt))
                tmp_mean.append(np.mean(T_crop_raw))
                tmp_peak_traction_filtered.append(np.max(T_filt))
                    
                # need to adjust crop to return only peak regions 
                # compare those (object detection), see TIRF-SIM paper
                tmp_svd_list.append(svd_metric(T_crop_raw,T_crop))
                tmp_peak_tractions.append(np.max(T_crop_raw))
                tmp_accuracy_list.append(accuracy_image(T_crop,T_crop_raw))
                tmp_precision_list.append(precision_image(T_crop,T_crop_raw))
                
                
                show_maps = False
                
                if show_maps == True: 
                    minmin = 0
                    maxmax = None
                    vmax = 1
                    cmap = None
                    
                    title =  str(f) + 'nN on ' + str(e) + ' kPa'
                    
                    fig, axes = plt.subplots(nrows=1, ncols=3, dpi = 150, figsize = (15,5))
    
                    im1 = axes[0].imshow(T, vmin=minmin, vmax=maxmax,
                              aspect='auto', cmap= cmap)
                    im2 = axes[1].imshow(T_filt, vmin=minmin, vmax=maxmax,
                             aspect='auto', cmap= cmap)
                    im3 = axes[2].imshow((T-T_filt), vmin=None, vmax= None, aspect ='auto', cmap= cmap)
    
                    fig.subplots_adjust(right=0.85)
                    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
                    cbar_ax2 = fig.add_axes([0.99, 0.15, 0.04, 0.7])
                    fig.colorbar(im2, cax=cbar_ax)
                    fig.colorbar(im3, cax = cbar_ax2)
                    fig.suptitle(title)
                    axes[0].set_title('raw')
                    axes[1].set_title('filtered')
                    axes[2].set_title('accuracy')
                    #path = 'plotting/traction_maps/displacement_' + str(i) + '.png'
                    #plt.savefig(path)
                    plt.show()
                    #print('max_traction_accuracy: ', (np.max(T)-np.max(T_filt))/np.max(T))
    
                show_hist = False
    
                if show_hist == True: 
                    plt.figure(dpi= 150)
                    plt.title(i)
                    bins = np.linspace(0, 6000, 100)
                    hist, bins = np.histogram(T, bins = bins)
                    hist_f, bins_f = np.histogram(T_filt, bins = bins )
                    
    
                    
                    plt.plot(bins[0:-1], hist ,color = 'blue', label = 'raw', alpha = .5)
                    plt.plot(bins[0:-1], hist_f, color = 'red', label = 'filtered', alpha = .5)
    
                    #plt.plot(bins[0:-1], (hist-hist_f)/max(hist), color = 'green', label = 'raw - filtered')
                    plt.xlabel('displacement [nm]')
                    plt.ylabel('frequency')
                    plt.xlim(0,1000)
                    #plt.ylim(0,1500)
                    plt.legend()
                    #plt.savefig('plotting/comparing_hist_'+ i[:-4] +  '.png')
                    plt.show()
                    #tmp_accuracy_list.append(np.median(T_crop/T_crop_raw))

            peak_traction_filtered.append(tmp_peak_traction_filtered)
            mean_traction_list.append(tmp_mean_traction_list)
            noise_list.append(tmp_noise_list)
            mean_raw.append(tmp_mean)
            svd_list.append(tmp_svd_list)
            accuracy_list.append(tmp_accuracy_list)
            precision_list.append(tmp_precision_list)
            
    def means(ls):
        new_ls =[]
        for j in range(len(ls[0])):
            tmp_ls = []
            for i in range(len(ls)):
                tmp_ls.append(ls[i][j])
            new_ls.append(np.mean(tmp_ls))
        return np.array(new_ls)
    
    # Put generated lists in correct hsape
    s = (int(len(force_list)),int(len(young_list)))
    noise_list = np.reshape(means(noise_list), s)
    mean_traction_list = np.reshape(means(mean_traction_list), s)
    peak_traction_filtered = np.reshape(means(peak_traction_filtered),s)
    mean_raw  = np.reshape(means(mean_raw),s)
    mean_svd = np.reshape(means(svd_list),s)
    mean_accuracy = np.reshape(means(accuracy_list),s)
    mean_precision = np.reshape(means(precision_list),s)
    vmax = None
    #%%
    # Plot of noise behaviour
    
    n = int(np.round(len(force_list)/7,0)) #spacing label
    if n == 0: 
        n = 1
        
    print('Noise matrix [Pa]: \n',noise_list)
    plt.figure(dpi = 150)
    plt.imshow(noise_list, origin = 'lower', vmax = np.max(noise_list))
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('noise $\sigma$ of displacements on gels')
    plt.colorbar()
    plt.show()

    plt.figure(dpi = 150)
    plt.imshow(mean_traction_list, origin = 'lower')
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('mean experienced displacements - recovered')
    plt.colorbar()
    plt.show()
    
    
    plt.figure(dpi = 150)
    plt.imshow(mean_raw, origin = 'lower') 
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('mean experienced displacements - raw')
    plt.colorbar()
    plt.show()
    
    plt.figure(dpi = 150)
    a = mean_traction_list /mean_raw 
    plt.imshow(np.logical_and(a>=.8, a<=1.2), origin = 'lower') 
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('recovery precision of mean displacement filtered/raw') # you could also call this accuracy.
    plt.colorbar()
    plt.show()
    
    
    plt.figure(dpi = 150)
    plt.imshow(peak_traction_filtered, origin = 'lower')
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('peak experienced displacements')
    plt.colorbar()
    plt.show()

    plt.figure(dpi = 150)
    plt.imshow(mean_traction_list/noise_list, origin = 'lower', cmap = None, vmax =None)#= np.max(noise_list))
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('mean displacement center / $\sigma$ noise')
    plt.colorbar()
    #plt.savefig('sigma_background_divided_by_max_traction.png')
    plt.show()
    
    #%%
    ratio = mean_traction_list / mean_raw
    new_ratio = ratio.copy()
    for i in range(len(ratio)):
        for j in range(len(ratio[i])):
            if ratio[i][j] < 0.8 or ratio[i][j] > 1.2:
                new_ratio[i][j] = 0
    
    norm_mean_svd = np.log(mean_svd.copy()) 
    norm_mean_svd = norm_mean_svd #/ np.linalg.norm(norm_mean_svd)        
    
    plt.figure(dpi = 150)
    plt.imshow(norm_mean_svd, vmin = None, vmax = None, origin = 'lower', cmap = 'viridis_r')
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('SVD score center crop')
    plt.colorbar()
    plt.savefig('plotting/svd.png')
    plt.show()
    
    plt.figure(dpi = 150)
    plt.imshow(mean_accuracy, vmin = 0, vmax = 1, origin = 'lower')
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('Accuracy score center crop')
    plt.colorbar()
    plt.savefig('plotting/accuracy.png')
    plt.show()
    
    plt.figure(dpi = 150)
    plt.imshow(mean_precision, vmin = 0, vmax = 1, origin = 'lower')
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('Precision score center crop')
    plt.colorbar()
    plt.savefig('plotting/precision.png')
    plt.show()
    
    #%%
    np.savez('results_confocal.npz', mean_svd, mean_accuracy, mean_precision)

    #%%
    # Calculate mean force heatmap to compare datasets

    xx, yy = np.meshgrid(force_list, young_list)
    # plt.scatter(xx,yy) 

    traction_map  = traction(force_list, young_list)
    filtered_traction_map = filtered_traction(force_list, young_list)
    #%%
    plt.figure( dpi = 150)
    plt.imshow(traction_map)
    plt.title('max traction')
    plt.colorbar()
    plt.show()

    plt.figure( dpi = 150)
    plt.title('filtered_max_traction')
    plt.imshow(filtered_traction_map)
    plt.colorbar()
    plt.show()

    print('Accuracy: ', np.max(filtered_traction_map)/np.max(traction_map)*100, '%s')

    plt.figure(dpi = 150)
    plt.imshow(filtered_traction_map/traction_map )
    plt.colorbar()
    plt.show()
    print(np.round(filtered_traction_map/traction_map, decimals= 4))
    #plt.xticks([round(i) for i in range(0,len(young_list),xstep)],young_list[::xstep], rotation = 90)
    #plt.yticks([round(i) for i in range(0,len(force_list),ystep)], np.round(force_list[::ystep], decimals= 2))


    #%%
    # plot force profiles 
    noise_list = []
    noise_list_raw = []

    mean_traction_list = []
    peak_traction_filtered = []
    accuracy_list = []
    for f in force_list:
        for e in young_list:
            path = 'tractions/tractions_displacement_' + str(f) + '_nN_on_' + str(e) +'_kPa.npz'
            path_filtered = 'tractions/tractions_filtered_displacement_' + str(f) + '_nN_on_' + str(e) +'_kPa.npz'
            T = np.load(path)['qAbs']
            #T_mask = (T < (0.04))
            #T = T_mask * T
            T_filt = np.load(path_filtered)['qAbs']
            i = 'tractions_' + str(f) + '_nN_on_' + str(e) +'_kPa.npz'
            
            # determine and append noise:
            # using skimage estimator:    
            #noise_list.append(2*estimate_sigma(T_filt))
            # using mean of a low impact window at upper right 80-90 % corner length:
            T_crop = T_filt[ int(0.8*len(T_filt)) : int(0.9*len(T_filt))
                            ,int(0.8*len(T_filt)) : int(0.9*len(T_filt))]
            
            T_crop_raw = T_filt[ int(0.8*len(T)) : int(0.9*len(T))
                            ,int(0.8*len(T)) : int(0.9*len(T))]
            
            noise_list.append(np.mean(T_crop))
            noise_list_raw.append(np.mean(T_crop_raw))
            plot_sub_image = False
            if plot_sub_image == True: 
                plt.imshow(T_crop, vmin = 0, vmax = 10, origin = 'lower', cmap = 'turbo')
                plt.colorbar()
                plt.show()
                #print(noise_list[-1])
            
            #print('bg_traction: ',noise[-1], ' Pa')
            # crop center to didtch edges
            T_crop_center = T_filt[ int(0.2*len(T_filt)) : int(0.8*len(T_filt))
                            ,int(0.2*len(T_filt)) : int(0.8*len(T_filt))]
            mean_traction_list.append(np.mean(T_crop_center))
            peak_traction_filtered.append(np.max(T_crop_center))
            
            title =  str(f) + 'nN on ' + str(e) + ' kPa'

            show_maps = True 
            if show_maps == True: 
                minmin = 0
                maxmax = None
                vmax = 1
                cmap = 'turbo'

                fig, axes = plt.subplots(nrows=1, ncols=3, dpi = 150, figsize = (15,5))

                im1 = axes[0].imshow(T, vmin=minmin, vmax=maxmax,
                        aspect='auto', cmap= cmap)
                im2 = axes[1].imshow(T_filt, vmin=minmin, vmax=maxmax,
                        aspect='auto', cmap= cmap)
                
                accuracy  = T_filt / T
        
                im3 = axes[2].imshow(accuracy, vmin=0, vmax= None, aspect ='auto', cmap= cmap)

                fig.subplots_adjust(right=0.85)
                cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
                cbar_ax2 = fig.add_axes([0.99, 0.15, 0.04, 0.7])
                fig.colorbar(im2, cax=cbar_ax)
                fig.colorbar(im3, cax = cbar_ax2)
                fig.suptitle(title)
                axes[0].set_title('raw')
                axes[1].set_title('filtered')
                axes[2].set_title('norm filtered - norm raw')
                path = 'plotting/traction_maps/' + str() + '.png'
                #plt.savefig(path)
                plt.show()
                #print('max_traction_accuracy: ', (np.max(T)-np.max(T_filt))/np.max(T))

            show_hist = False

            if show_hist == True: 
                plt.figure(dpi= 150)
                plt.title(i)
                bins = np.linspace(0, 6000, 100)
                hist, bins = np.histogram(T, bins = bins)
                hist_f, bins_f = np.histogram(T_filt, bins = bins )
                

                
                plt.plot(bins[0:-1], hist ,color = 'blue', label = 'raw', alpha = .5)
                plt.plot(bins[0:-1], hist_f, color = 'red', label = 'filtered', alpha = .5)

                #plt.plot(bins[0:-1], (hist-hist_f)/max(hist), color = 'green', label = 'raw - filtered')
                plt.xlabel('traction [Pa]')
                plt.ylabel('frequency')
                #plt.xlim(0, 0.005)
                #plt.ylim(0,1500)
                plt.legend()
                #plt.savefig('plotting/comparing_hist_'+ i[:-4] +  '.png')
                plt.show()

    # %%
    # Plot line profiles
    plt.figure(dpi = 150)
    plt.title('background tractions for forces across gels')
    plt.plot(np.transpose(noise_list),label =  force_list)
    plt.legend()
    plt.ylabel('T [PA]')
    plt.xlabel('E [kPa]')
    plt.xticks([i for i in range(len(young_list))], young_list)
    plt.show()

    plt.figure(dpi = 150)
    plt.title('background tractions for gels')
    plt.plot(noise_list,label =  young_list)
    plt.legend()
    plt.xticks([i for i in range(len(force_list))], force_list)
    plt.xlabel('F [nN]')
    plt.ylabel('T [PA]')

    plt.show()

    plt.figure(dpi = 150)
    plt.title('background tractions for forces across gels')
    plt.plot(np.transpose(noise_list),label =  force_list)
    plt.legend()
    plt.ylabel('T [PA]')
    plt.xlabel('E [kPa]')
    plt.xticks([i for i in range(len(young_list))], young_list)
    plt.show()
    #%%
    # Plot signal to Noise
    n = int(np.round(len(force_list)/7,0)) #spacing label
    if n == 0: 
        n = 1
    plt.figure(dpi = 150)#0, figsize=(10,10))
    plt.title('Signal / noise')
    plt.imshow((peak_traction_filtered/(noise_list)),vmax = None, origin ='lower')#, norm = LogNorm(vmin = 1, vmax = 10e3))
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('Signal to Noise ratio')
    clb = plt.colorbar()
    clb.ax.set_ylabel('T [Pa]', rotation = -90)
    plt.show()

    #%%
    plt.figure(dpi = 150)
    plt.imshow((peak_traction_filtered/(noise_list))>5,vmax = None, origin ='lower')
    plt.xticks([i for i in range(len(young_list))][::n], np.round(young_list[::n],1))
    plt.yticks([i for i in range(len(force_list))][::n], np.round(force_list[::n],2))
    plt.xlabel('E [kPa]')
    plt.ylabel('F [nN]')
    plt.title('Resolvability SNR > 5')
    plt.colorbar()
    plt.show()
    #%%
    x = []
    std = []
    for i in np.transpose(noise_list): 
        x.append(np.mean(i))
        std.append(np.std(i))

    fig, axes = plt.subplots(dpi = 150)
    plt.plot(young_list, x, c = 'b')
    low = np.array(x) - np.array(std)
    up  = np.array(x) + np.array(std)
    axes.fill_between(np.array(young_list), low, up, facecolor = 'blue', alpha = .25)
    axes.plot(young_list, np.transpose(noise_list), alpha = 0.2, c ='black' )

    #plt.ylim(0,500)
    axes.set_xlabel('E [kPa]')
    axes.set_ylabel('T [Pa]')
    axes.set_title('background tractions for gels')
    plt.show()
    #%%
    # plot force projections for desired stiffness
    E = young_list[1] 
    print('Chosen Stiffness: ',E, ' kPa')
    f_projection = []
    for F in force_list:
        path = 'tractions/tractions_filtered_displacement_' + str(F) + '_nN_on_' + str(E) + '_kPa.npz'
        npz = np.load(path)
        tractions = npz['qAbs'] 
        f_projection.append(tractions)
        
        # plt.imshow(tractions, vmax = 5000)
        # plt.colorbar()
        # plt.show()
        
        
    napari.view_image(np.array(f_projection), colormap = 'turbo')
    #%%
    # Current adjustments: get  filter to work! 
    # Best do it in  the following way: 
    # - load npz file of interest. 
    # - copy variable
    # - rewrite it according to filter (set all displacements under threshhold to 0)
    # - save in new folder (filtered_displacement)
    # - make alternative folder callable. in TFM reconstruct. 
    # --> enjoy the finished skript / project
    #
    # Done ! filter now works simulation is callable. 
    # 
    #
    # Arising issue: 
    # The native runtfm.py only exports graphs. 
    # --> need to export the reconstructed datapoints for proper custom analysis. 
    # This means we need to insert a code sniplet into runtfm.py that exports the data into a npz file 
    # OR which is way better, import run_FTTC3d function like we did in the other skript. 
    # This npz file will then get picked up by this skript, renamed and moved to an appropriate location. 
    # Analyse the available data in a phase plot (whatever that means)
    # 
    # Suggested analysis: 
    # compare force distribution between the ground "truth" and the filtered data, using a histogram. 
    # --> arising problem: many different graphs, and only suttle differences. 
    # available parameters: 
    # - Stiffness / Youngs modulus 
    # - Force intensity 
    # - calculated force 
    # - standard deviation of force. 
    # - raw vs filtered 

    # 1) Plot a map of stiffness youngs modulus and mean forces (meshgrid and image)
    # 2) Compare the two maps for filtered vs non filtered. 
    # 
    # Alternatively: plot deviation of youngs modulus instead of mean modulus; or plot median modulus.
    # Hypothesis: displacement distributions change according to optical system therefore the reconstructed forces should change as well. 
    # Is this relevant in our sprectra / parameterspace.


    # UPDATE: 
    # checking the documentation of the Blumberg skript revealed that the build in noise function puts a gaussian blur 
    # onto the displacement field. This is exactly the same the skript provides here
    # But: our skript here provides more utility in terms of filtering and adjustability. 
    # --> what is an appropriate way to filter the available data to the presented question?
    # --> need for a clean formulation of the presented question.

    # question: Is mechanosensation real, or only a result of force reconstruction with unsufficient microscope resolution?
    # --> defining factor for displacement field resolution --> bead density 
    # --> what influence has microscope resolution onto our system ? 
    # --> theoretically little to none, because sub pixel resolution
    # --> under the assumption of perfect bead distribution what does increase of resolution allow? 
    # --> only advantage: distinguish beads from each other, higher resolved displacement field.
    # --> rephrase the question: has the dynamic range of recorded displacement an influence on resolved tractions?
    # --> high displacements are rare but more relevant.
    # --> how to define upper limit of the dynamic range? --> currently 1um max displacement, but how to replace?
    # --> mask original image and interpolate max values.
    # --> how to precicely interpolate only for these values? --> keep changelog?
    # --> or set them as 1 because max, displacement is reached?
    # --> How does noise on marker translate to noise in displacement field 
    # --> noise shape is the same; but is the amplitude the same? 
    # --> how do we get from microscope resolution error to displacement field error

    return

# Run Skript
if __name__ == "__main__":
    main()