import building_volume_frac
import numpy as np
import time 


#print(decomp.python_set_up_recbis.__doc__)

t0 = time.time()

print(t0)

#nx=200
#ny=100
nx=9600
ny=9600
nonods=nx*ny
#x = np.zeros(nonods)
#y = np.zeros(nonods)
xy = np.zeros((nonods,2))
#vol_frac = np.zeros(nonods)
boarders_size = np.zeros(3)
boarders_size[0]=0.06 
boarders_size[1]=0.06
boarders_size[2]=0.12

print("start running")

for j in range(ny): 
   for i in range(nx): 
      nod= i + j*nx
      xy[nod,0] = (np.float(i-1)/np.float(nx-1) ) * 4.0 # length of domain is 2.0 
      xy[nod,1] = (np.float(j-1)/np.float(ny-1) ) * 4.0 # width of domain is 2.0 

min_dx = 0.01 # Min building size.
max_dx = 0.08 # Max building size.
size_gap = 0.08 # Gap between buildings. 
nbuilding = 2000 # no of buildings to aim for
nit_build = 1000 # no of times to try to produce a building
nx_fix = 400 # no of cells in x-direction of the fixed mesh used for generating buildings in buildings code. 
ny_fix = 400 # no of cells in y-direction of the fixed mesh used for generating buildings in buildings code. 

#print "split_levels", split_levels
print (building_volume_frac.python_building_vol_frac.__doc__)
# we have to finish the sub clal with the variables used in the decleration within this sub.
#witchd=decomp_new.python_set_up_recbis(split_levels,fina,cola, wnod,a, havwnod,havmat,iexact, nsplt,ncola,n,na)
#vol_frac,nbuild = building_volume_frac.python_building_vol_frac(x,y,min_dx,max_dx,size_gap, boarders_size, nbuilding, nit_build, nonods) 
vol_frac,nbuild = building_volume_frac.python_building_vol_frac(xy,min_dx,max_dx,size_gap, boarders_size, nbuilding, nit_build, nx_fix, ny_fix, nonods)  
#print(witchd)

for j in range(ny): #do j=1,ny
    print (vol_frac[j*nx : (j+1)*nx])

vol_frac_plot = vol_frac.reshape((ny,nx))

print("Training time:", time.time()-t0)
np.save('./fortran_building_volume_frac_codes/vol_frac_plot_{}_{}_buildings{}_mindx{}_maxdx{}__new1.npy'.format(nx,ny,nbuilding, int(min_dx*100), int(max_dx*100)), vol_frac_plot)
print("Finished")
