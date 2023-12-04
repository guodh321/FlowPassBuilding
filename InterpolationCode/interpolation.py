import numpy as np
import sys, os
import u2r
import vtk, vtktools

#from collections import OrderedDict


field_name = 'Velocity'

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
for i in range(0,399):
    # filename = '/home/dg321/gitTest/PRI/irp/Flow_Data/RawData/Flow_past_buildings_' + str(i) + '.vtu'  # for 6*6
    filename = '/home/dg321/gitTest/PRI/irp/Flow_Data_9_9/RawData/Flow_past_buildings_' + str(i) + '.vtu'  # for 9*9
    vtu_data =  vtktools.vtu(filename)
    coordinates = vtu_data.GetLocations()
    nNodes = coordinates.shape[0] 
    nEl = vtu_data.ugrid.GetNumberOfCells()
    print('nNodes:', nNodes, ', nEl:', nEl) 


    # code works for reading in one time level of one field
    # VelocityAbsorption is building
    
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
    #x_ndgln = np.zeros((nEl*nloc), dtype=int)
    x_ndgln = np.zeros((nEl*nloc), dtype=int)
    for iEl in range(nEl):
        n = vtu_data.GetCellPoints(iEl) + 1
        x_ndgln[iEl*nloc:(iEl+1)*nloc] = n


    # set grid size
    ddx_size = ndim
    #nx = 128   #I chose these values because it's easier for the CNN later.
    #ny = 128
    #nz = 32
    nGrid = np.zeros((ndim),dtype=int)
    
    # nGrid[0] = 256 # nx  for 6*6
    # nGrid[1] = 256 # ny  for 6*6

    nGrid[0] = 384 # nx  for 9*9
    nGrid[1] = 384 # ny  for 9*9

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

    print('')
    zeros_on_mesh = 0
    print("Value_mesh:    ", value_mesh.shape)
    print("x_all:         ", x_all.shape)
    print("x_ndgln:       ", x_ndgln.shape)
    print("ddx:           ", ddx.shape)
    print("block_x_start: ", block_x_start.shape)
    print("nx, ny, nz:    ", nx, ny, nz)
    print("zeros_on_mesh: ", zeros_on_mesh)
    print("nEl:           ", nEl)
    print("nloc:          ", nloc)
    print("nNodes:        ", nNodes)
    print("nscalar:       ", nscalar)
    print("ndim:          ", ndim)
    print("nTime:         ", nTime)

    # new variable!
    nElnloc = nEl*nloc

    # both calls to u2r work for me
    zeros_on_mesh = 0
    value_grid = u2r.simple_interpolate_from_mesh_to_grid(value_mesh,x_all,x_ndgln,ddx,block_x_start,nx,ny,nz,zeros_on_mesh,nEl,nloc,nElnloc,nNodes,nscalar,ndim,nTime)

    print("valueeeee:", value_grid.shape)
    print('np.max(value_grid)', np.max(value_grid))
    print('np.min(value_grid)', np.min(value_grid))

    # np.save('/home/dg321/gitTest/PRI/irp/Flow_Data/InterpolatedResult1024/FpB_Interpolated_t{}_{}_{}_{}'.format(i, field_name, nx, ny), value_grid[:,:,:,0,0])
    # np.save('/home/dg321/gitTest/PRI/irp/Flow_Data/InterpolatedResult256/FpB_Interpolated_t0_VelocityAbsorption_256_256.npy', value_grid[:,:,:,0,0])  # for 6*6

    np.save('/home/dg321/gitTest/PRI/irp/Flow_Data_9_9/FpB_Interpolated_t{}_{}_{}_{}'.format(i, field_name, nx, ny), value_grid[:,:,:,0,0])  # for 9*9

    print('Finished: FpB_Interpolated_t{}_{}_{}_{}.'.format(i, field_name, nx, ny))

