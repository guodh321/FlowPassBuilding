import numpy as np
import sys, os
sys.path.append("..")
sys.path.append('/home/cheaney/Code/interpolation')
import u2r
sys.path.append('/usr/lib/python3/dist-packages/')
from tools import vtktools
import vtk


#from collections import OrderedDict

def get_clean_vtk_file(filename):
    "Removes fields and arrays from a vtk file, leaving the coordinates/connectivity information."
    vtu_data = vtktools.vtu(filename)
    clean_vtu = vtktools.vtu()
    clean_vtu.ugrid.DeepCopy(vtu_data.ugrid)
    fieldNames = clean_vtu.GetFieldNames()
# remove all fields and arrays from this vtu
    for field in fieldNames:
        clean_vtu.RemoveField(field)
        fieldNames = clean_vtu.GetFieldNames()
        vtkdata=clean_vtu.ugrid.GetCellData()
        arrayNames = [vtkdata.GetArrayName(i) for i in range(vtkdata.GetNumberOfArrays())]
    for array in arrayNames:
        vtkdata.RemoveArray(array)
    return clean_vtu

# after importing the u2r library you can get some information using these commands:
# help(u2r)
print(u2r.simple_interpolate_from_mesh_to_grid.__doc__)

###
### read in data from the vtu file (assuming DG fields)
###

i=499
filename = 'Flow_past_buildings_' + str(i) + '.vtu'
vtu_data =  vtktools.vtu(filename)
coordinates = vtu_data.GetLocations()
nNodes = coordinates.shape[0] 
nEl = vtu_data.ugrid.GetNumberOfCells()
print('nNodes:', nNodes, ', nEl:', nEl) 


# code works for reading in one time level of one field
field_name = 'Velocity'
field_values = [] # no need to save as a list, although I did this here as sometimes we have more than one field
field_values.append(vtu_data.GetField(field_name))
nTime = 1 # reading in just one file


# ndim and ndim_field could be different (eg a 2D problem (ndim=2) and reading in a temperature field (ndim_field=1))
ndim = 2 # dimension of physical problem (ie the coordinates)
ndim_field = 2 # dimension of the field (scalar, vector, tensor field)
nloc = 3 # number of nodes per tetrahedral element (for 2D nloc = 3, for 3D nloc = 4)  
nscalar = ndim_field # when reading in more than one field nscalar \neq ndim_field
# paraview writes 2D problems to 3D meshes, but with zeros in the third direction - we don't need the zeros
x_all = np.transpose(coordinates[:,0:ndim]) 


print('ndim_field or nscalar',nscalar)
value_mesh = np.zeros((nscalar,nNodes,nTime))
print('value_mesh[0,:,0].size',value_mesh[0,:,0].shape)
print('field_values[:,0].size',field_values[0].shape)
value_mesh[0,:,0] = field_values[0][:,0]
for ii in range(ndim_field):
    value_mesh[ii,:,0] = field_values[0][:,ii] # first zero as nTime=1, second zero as we only read in one field

###
### get arrays required to interpolate to the grid and set grid size
###

# rectangular domain so
xmin = np.zeros((ndim))
xmax = np.zeros((ndim))
for ii in range(ndim):
    xmin[ii] = min(coordinates[:,ii])
    xmax[ii] = max(coordinates[:,ii])
print('(xmin)',xmin)
print('(xmax)',xmax)
#block_x_start = np.array(( x0, y0, z0 )) 
block_x_start = np.array(( xmin )) 

# get global node numbers - I think this is only for a DG mesh
x_ndgln = np.zeros((nEl*nloc), dtype=int)
print(x_ndgln.shape)

# x_ndgln = np.zeros((nEl*nloc), dtype=int)
for iEl in range(nEl):
    n = vtu_data.GetCellPoints(iEl) + 1
    x_ndgln[iEl*nloc:(iEl+1)*nloc] = n


# set grid size
ddx_size = ndim
#nx = 128#I chose these values because it's easier for the CNN later.
#ny = 128
#nz = 32
nGrid = np.zeros((ndim),dtype=int)
nGrid[0] = 101 # nx
nGrid[1] = 101 # ny

ddx = np.zeros((ddx_size))
#for ii in range(ndim):
#    ddx[ii] = (xmax[ii]-xmin[ii])/(nGrid[ii]-1) 
ddx = np.divide( xmax-xmin, nGrid-1 ) 
print('ddx:',ddx.shape,ddx)

print('domain lengths:', xmax - xmin)
print('grid lengths:',   np.multiply(nGrid-np.ones(ndim),ddx )) #(nx-1)*ddx[0], (ny-1)*ddx[1])#, (nz-1)*ddx[2])
print(' ')
print('np.max(value_mesh):', np.max(value_mesh))
print('np.min(value_mesh):', np.min(value_mesh))

nx = nGrid[0]
ny = nGrid[1]
nz = 1
if ndim ==3:  
    nz = nGrid[2]

###
### interpolate from (unstructured) mesh to (structured) grid
###

# x_ndgln = x_ndgln.reshape(nEl*nloc, 1)
print(x_ndgln.shape)

nEl=53533
nloc=3
nscalar=2,
ndim=2,
nTime=1

# print(shape(x_ndgln, 0))

zeros_on_mesh = 0
value_grid = u2r.simple_interpolate_from_mesh_to_grid(value_mesh,x_all,ddx,block_x_start,nx,ny,nz,zeros_on_mesh,nEl,nloc,nNodes,nscalar,ndim,nTime)
# value_grid = simple_interpolate_from_mesh_to_grid(value_mesh,x_all,x_ndgln,ddx,block_x_start,nx,ny,nz,ireturn_zeros_outside_grid,[totele,nloc,nonods,nscalar,ndim,ntime]) 

print('np.max(value_grid)', np.max(value_grid))
print('np.min(value_grid)', np.min(value_grid))

# Donghu: value_grid is the solution from the mesh interpolated onto a regular grid 101 by 101 - you might
# want to use numbers that are more amenable for a CNN, i.e. 2^N by 2^N

###
### remesh, i.e. interpolate from (structured) grid back to the (unstructured) mesh 
###

zeros_on_grid = 1
value_remesh = u2r.interpolate_from_grid_to_mesh(value_grid, block_x_start, ddx, x_all, zeros_on_grid, nscalar, nx, ny, nz, nNodes, ndim, nTime)

print('np.max(value_mesh):', np.max(value_remesh))
print('np.min(value_mesh):', np.min(value_remesh))
print('')


###
### write the remeshed results to file to compare with the original results
###

pointwise_error = value_remesh - value_mesh

print('Writing vtu file ...')

# get the mesh connectivity and coords
filename = 'Flow_past_buildings_' + str(i) + '.vtu'
clean_vtu = get_clean_vtk_file(filename)

# start making a new file
filename = 'Flow_past_buildings_REMESH_' + str(i) + '.vtu'
new_vtu = vtktools.vtu()
new_vtu.ugrid.DeepCopy(clean_vtu.ugrid)
new_vtu.filename = filename
if ndim_field<3:
    #print(np.squeeze(value_remesh[0:ndim_field,:,:]).T.shape)
    new_vtu.AddField(field_name, np.squeeze(value_remesh[0:ndim_field,:,:]).T)
    new_vtu.AddField('pointwise error', np.squeeze(pointwise_error[0:ndim_field,:,:]).T)    
elif ndim_field==3: # not sure if this works / if it nees to be different from the above
    my_array = value_remesh[0:ndim_field]
    print('my array (ndims==3)', my_array.shape, np.squeeze(my_array).T.shape) 
    new_vtu.AddField(field_name, np.squeeze(my_array).T)
    new_vtu.AddField('pointwise error', np.squeeze(my_array).T)
else:
    print('ndim_field value not expected (should be either 2 or 3...)')
    sys.exit() 

new_vtu.Write()
print('Finished.')

