import numpy as np
from numpy import ravel
#from scipy.ndimage import label, binary_fill_holes
#from skimage.measure import perimeter, perimeter_crofton, regionprops, regionprops_table
from .cpm import CPM

import numba
from numba import jit



############## NOT MY CODE ######################################

def perimeter_crofton_source_code(image, directions=4):
    if image.ndim != 2:
        raise NotImplementedError('`perimeter_crofton` supports 2D images only')

    # as image could be a label image, transform it to binary image
    image = (image > 0).astype(np.uint8)
    image = np.pad(image, pad_width=1, mode='constant')
    XF = convolve(
        image, np.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]]), mode='constant', cval=0
    )

    XF = np.asarray(XF)
    h = np.bincount(XF.ravel(), minlength=16)

    # definition of the LUT
    if directions == 2:
        coefs = [
            0,
            np.pi / 2,
            0,
            0,
            0,
            np.pi / 2,
            0,
            0,
            np.pi / 2,
            np.pi,
            0,
            0,
            np.pi / 2,
            np.pi,
            0,
            0,
        ]
    else:
        coefs = [
            0,
            np.pi / 4 * (1 + 1 / (np.sqrt(2))),
            np.pi / (4 * np.sqrt(2)),
            np.pi / (2 * np.sqrt(2)),
            0,
            np.pi / 4 * (1 + 1 / (np.sqrt(2))),
            0,
            np.pi / (4 * np.sqrt(2)),
            np.pi / 4,
            np.pi / 2,
            np.pi / (4 * np.sqrt(2)),
            np.pi / (4 * np.sqrt(2)),
            np.pi / 4,
            np.pi / 2,
            0,
            0,
        ]

    total_perimeter = coefs @ h
    return total_perimeter

def _get_output(output, input, shape=None, complex_output=False):
    if shape is None:
        shape = input.shape
    if output is None:
        if not complex_output:
            output = np.zeros(shape, dtype=input.dtype.name)
        else:
            complex_type = np.promote_types(input.dtype, np.complex64)
            output = np.zeros(shape, dtype=complex_type)
    elif isinstance(output, type | np.dtype):
        # Classes (like `np.float32`) and dtypes are interpreted as dtype
        if complex_output and np.dtype(output).kind != 'c':
            #warnings.warn("promoting specified output dtype to complex", stacklevel=3)
            output = np.promote_types(output, np.complex64)
        output = np.zeros(shape, dtype=output)
    elif isinstance(output, str):
        output = np.dtype(output)
        if complex_output and output.kind != 'c':
            raise RuntimeError("output must have complex dtype")
        elif not issubclass(output.type, np.number):
            raise RuntimeError("output must have numeric dtype")
        output = np.zeros(shape, dtype=output)
    else:
        # output was supplied as an array
        output = np.asarray(output)
        if output.shape != shape:
            raise RuntimeError("output shape not correct")
        elif complex_output and output.dtype.kind != 'c':
            raise RuntimeError("output must have complex dtype")
    return output

def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input and complex_weights:
        # real component of the output
        func(input.real, weights.real, output=output.real,
             cval=np.real(cval), **kwargs)
        output.real -= func(input.imag, weights.imag, output=None,
                            cval=np.imag(cval), **kwargs)
        # imaginary component of the output
        func(input.real, weights.imag, output=output.imag,
             cval=np.real(cval), **kwargs)
        output.imag += func(input.imag, weights.real, output=None,
                            cval=np.imag(cval), **kwargs)
    elif complex_input:
        func(input.real, weights, output=output.real, cval=np.real(cval),
             **kwargs)
        func(input.imag, weights, output=output.imag, cval=np.imag(cval),
             **kwargs)
    else:
        if np.iscomplexobj(cval):
            raise ValueError("Cannot provide a complex-valued cval when the "
                             "input is real.")
        func(input, weights.real, output=output.real, cval=cval, **kwargs)
        func(input, weights.imag, output=output.imag, cval=cval, **kwargs)
    return output

def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution, axes):
    input = np.asarray(input)
    weights = np.asarray(weights)
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input or complex_weights:
        if complex_weights and not convolution:
            # As for np.correlate, conjugate weights rather than input.
            weights = weights.conj()
        kwargs = dict(
            mode=mode, origin=origin, convolution=convolution, axes=axes
        )
        output = _get_output(output, input, complex_output=True)

        return _complex_via_real_components(_correlate_or_convolve, input,
                                            weights, output, cval, **kwargs)



def convolve(input, weights, output=None, mode='reflect', cval=0.0,
             origin=0, *, axes=None):
    return _correlate_or_convolve(input, weights, output, mode, cval,
                                  origin, True, axes)

def _binary_erosion(input, structure, iterations, mask, output,
                    border_value, origin, invert, brute_force, axes):
    try:
        iterations = operator.index(iterations)
    except TypeError as e:
        raise TypeError('iterations parameter should be an integer') from e

    input = np.asarray(input)
    # The Cython code can't cope with broadcasted inputs
    if not input.flags.c_contiguous and not input.flags.f_contiguous:
        input = np.ascontiguousarray(input)

    ndim = input.ndim
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    if structure is None:
        structure = generate_binary_structure(num_axes, 1)
    else:
        structure = np.asarray(structure, dtype=bool)
    if ndim > num_axes:
        structure = _filters._expand_footprint(ndim, axes, structure,
                                               footprint_name="structure")

    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have same dimensionality')
    if not structure.flags.contiguous:
        structure = structure.copy()
    if structure.size < 1:
        raise RuntimeError('structure must not be empty')
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != input.shape:
            raise RuntimeError('mask and input must have equal sizes')
    origin = _ni_support._normalize_sequence(origin, num_axes)
    origin = _filters._expand_origin(ndim, axes, origin)
    cit = _center_is_true(structure, origin)
    if isinstance(output, np.ndarray):
        if np.iscomplexobj(output):
            raise TypeError('Complex output type not supported')
    else:
        output = bool
    output = _ni_support._get_output(output, input)
    temp_needed = np.may_share_memory(input, output)
    if temp_needed:
        # input and output arrays cannot share memory
        temp = output
        output = _ni_support._get_output(output.dtype, input)
    if iterations == 1:
        _nd_image.binary_erosion(input, structure, mask, output,
                                 border_value, origin, invert, cit, 0)
    elif cit and not brute_force:
        changed, coordinate_list = _nd_image.binary_erosion(
            input, structure, mask, output,
            border_value, origin, invert, cit, 1)
        structure = structure[tuple([slice(None, None, -1)] *
                                    structure.ndim)]
        for ii in range(len(origin)):
            origin[ii] = -origin[ii]
            if not structure.shape[ii] & 1:
                origin[ii] -= 1
        if mask is not None:
            mask = np.asarray(mask, dtype=np.int8)
        if not structure.flags.contiguous:
            structure = structure.copy()
        _nd_image.binary_erosion2(output, structure, mask, iterations - 1,
                                  origin, invert, coordinate_list)
    else:
        tmp_in = np.empty_like(input, dtype=bool)
        tmp_out = output
        if iterations >= 1 and not iterations & 1:
            tmp_in, tmp_out = tmp_out, tmp_in
        changed = _nd_image.binary_erosion(
            input, structure, mask, tmp_out,
            border_value, origin, invert, cit, 0)
        ii = 1
        while ii < iterations or (iterations < 1 and changed):
            tmp_in, tmp_out = tmp_out, tmp_in
            changed = _nd_image.binary_erosion(
                tmp_in, structure, mask, tmp_out,
                border_value, origin, invert, cit, 0)
            ii += 1
    if temp_needed:
        temp[...] = output
        output = temp
    return output

def binary_dilation(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0,
                    brute_force=False, *, axes=None):
    input = np.asarray(input)
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    if structure is None:
        structure = generate_binary_structure(num_axes, 1)
    origin = _ni_support._normalize_sequence(origin, num_axes)
    structure = np.asarray(structure)
    structure = structure[tuple([slice(None, None, -1)] *
                                structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1

    return _binary_erosion(input, structure, iterations, mask,
                           output, border_value, origin, 1, brute_force, axes)

def binary_fill_holes(input, structure=None, output=None, origin=0, *,
                      axes=None):

    input = np.asarray(input)
    mask = np.logical_not(input)
    tmp = np.zeros(mask.shape, bool)
    inplace = isinstance(output, np.ndarray)
    if inplace:
        binary_dilation(tmp, structure, -1, mask, output, 1, origin, axes=axes)
        np.logical_not(output, output)
    else:
        output = binary_dilation(tmp, structure, -1, mask, None, 1,
                                 origin, axes=axes)
        np.logical_not(output, output)
        return output


###################### enf of not my code #####################################


# updated to be skimage.measure.regionprops() perimeter
def _calculate_perimeter(cpm: CPM, cell_id):

    # skimage.measure.regionprops() perimeter
    binary_grid = (cpm.grid == cell_id)
    perimeter_value = perimeter_crofton_source_code(binary_grid, directions=4)

    return perimeter_value

def _fraction_illuminated(cpm: CPM, cell_id):
    
    """
    props = regionprops(label_image=cpm.grid, intensity_image=cpm.light_pattern)
    region = next((r for r in props if r.label == cell_id), None)
    
    if region is not None:
        return region.intensity_mean
    else:
        return 0.0
    """
    
    # old implementation:
    cell_mask = (cpm.grid == cell_id) # t/f mask of cell location
    light_mask = (cpm.light_pattern == 1) # t/f mask of light location
        
    overlap = cell_mask & light_mask # AND of both masks, where both true

    area_in_light = np.sum(overlap)
    total_area = np.sum(cell_mask)
        
    if total_area == 0:
        return 0.0
    return area_in_light / total_area


def _cell_contains_holes(cpm: CPM, cell_id):
    
    """
    cell_mask = (cpm.grid == cell_id).astype(np.uint8)  # binary mask for the cell
    
    # check that the cell has only one connected component
    labeled_array, num_features = label(cell_mask) # possibly: intepreter misreading the type of self.grid, totally fine at runtime
    assert num_features == 1, f"Expected exactly 1 connnected component, found {num_features}"

    # get region properties (of interest: euler number)
    props = regionprops(cell_mask)
    region = props[0]  # only one region
    
    return region.euler_number != 1 # if no holes, euler_number = 1 

    """
    # old implementation
   
    cell_mask = (cpm.grid == cell_id) # binary mask for the cell
    filled_mask = binary_fill_holes(cell_mask) # fill holes in the cell mask

    # compare original and filled masks - if equal, there are no holes
    contains_holes = not np.array_equal(cell_mask, filled_mask) # intepreter misreading the type of filled_mask, totally fine at runtime
    return contains_holes


###### HAMILTONIAN FUNCTION ###### ==> DIFFERENT HAMILTONIANS
    
def calculate_hamiltonian(cpm: CPM):
    """
    Compute the total Hamiltonian energy of the current CPM grid state.

    This Hamiltonian includes the following energy contributions for each cell:
    - Area deviation from the target area.
    - Deviation of the perimeter-to-area ratio from the target ratio.
    - Light consideration; energy of the cell is reduced if in an illuminated regions.

    Parameters
        cpm : CPM - CPM object as defined in cpm.py

    Returns
        hamiltonian : float - computed Hamiltonian energy value
            Returns np.inf if a cell is disjoint or contains holes.
    """
    
    hamiltonian = 0
    cell_ids = np.unique(cpm.grid)
    cell_ids = cell_ids[cell_ids != 0]

    # new implementation

    """
    properties = ['area','perimeter_crofton','intensity_mean', 'euler_number']
    props_table = regionprops_table(label_image=cpm.grid, intensity_image=cpm.light_pattern, properties=properties)

    if np.sum((props_table['euler_number'] != 1) > 0):
        return np.inf

    hamiltonian += np.sum(10*np.power(np.abs(props_table['area'] - cpm.target_area), 2))
    hamiltonian += np.sum(np.power(np.abs(props_table['perimeter_crofton']-cpm.target_perimeter), 4))
    hamiltonian -= np.sum(np.power(100 * np.abs(props_table['intensity_mean']), 3.5)) #1.75

    return hamiltonian
    """

    # old implementation
    
    for cell_id in cell_ids:
        # deltaH_ground: check for disjoint parts
        labeled_array, num_features = label(cpm.grid == cell_id) # intepreter misreading the type of self.grid, totally fine at runtime
        if num_features > 1:
            return np.inf  # positive infinity for disjoint parts
            
        #deltaH_ground: check for holes
        if _cell_contains_holes(cpm, cell_id):
            return np.inf
            
        # calc area & perimeter
        area = np.sum(cpm.grid == cell_id)
        perimeter = _calculate_perimeter(cpm, cell_id)

        # Energy terms for area and perimeter/area ratio
        hamiltonian += 10*np.power(np.abs(area - cpm.target_area), 2) # deltaH_area
        hamiltonian += np.power(np.abs(perimeter-cpm.target_perimeter), 4) # deltaH_perimeter
        hamiltonian += 0.8*(np.abs(((area**(1/2)) / perimeter) - cpm.target_ratio)) # deltaH_area/perimeter_ratio
        hamiltonian -= np.power(100*_fraction_illuminated(cpm, cell_id), 1.75) # no specific deltaH term as outlined in JP, but deltaH_lum for now
        
        # print statements
        #print("Cell ID: ", cell_id)
        #print("area delta: ", (area - cpm.target_area))
        #print("perimeter, perimeter delta: ", perimeter, (perimeter - cpm.target_perimeter))

    return hamiltonian

    