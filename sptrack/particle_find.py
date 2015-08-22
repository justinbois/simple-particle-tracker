import numpy as np
import scipy.optimize
import scipy.ndimage
import skimage
import skimage.io
import skimage.feature
import skimage.filters
import skimage.morphology
import pandas as pd


#%%
def preprocess(im, blur=True, boxcar=True, noise_size=1, boxcar_width=5):
    """
    Perform bluring and background subtraction of image.

    Parameters
    ----------
    im : ndarray
        Image to preprocess.
    blur : bool
        If True, perform a Gaussian blur on each image.
    boxcar : bool
        If True, perfrom a boxcar (mean) filter to compute background.
        The background is then subtracted.
    noise_size : float
        The characteristic length scale of noise in the images
        in units of pixels.  This is used to set the sigma value
        in the Gaussian blur.  Ignored if blur is False.
    boxcar_width : int
        Width of the boxcar filter.  Should be an odd integer greater
        than the pixel radius, but smaller than interparticle distance.

    Returns
    -------
    output : ndarray, shape as im, dtype float
        Blurred, background subtracted image.
    """

    # Convert the image to float
    im = skimage.img_as_float(im)

    # Return image back if we do nothing
    if not blur and not boxcar:
        return skimage.img_as_float(im)

    # Compute background using boxcar (mean) filter
    if boxcar:
        if boxcar_width % 2 == 0:
            raise ValueError('boxcar_width must be odd.')

        # Perform mean filter with a square structuring element
        im_bg = scipy.ndimage.uniform_filter(im, boxcar_width)
    else:
        im_bg = 0.0

    # Perform Gaussian blur
    if blur:
        im_blur = skimage.filters.gaussian_filter(im, noise_size)
    else:
        im_blur = im

    # Subtract background from blurred image
    bg_subtracted_image = im_blur - im_bg

    # Set negative values to zero and return
    return np.maximum(bg_subtracted_image, 0)


#%%
def local_maxima_pixel(im, particle_size, thresh_perc=70):
    """
    Find local maxima in pixel intensity.  Designed for use
    with images containing particles of a given size.

    Parameters
    ----------
    im : array_like
        Image in which to find local maxima.
    particle_size : float
        Diameter of particles in units of pixels.
    thresh_perc : float in range of [0, 100]
        Only pixels with intensities above the thresh_perc
        percentile are considered.  Default = 70.

    Returns
    -------
    output: ndarray
        (row, column) coordinates of peaks.
    """
    # Determine threshold for peak identification
    thresh = np.percentile(im, thresh_perc)

    # The minimum distance between peaks is a particle diameter
    return skimage.feature.peak_local_max(
        im, min_distance=particle_size, threshold_abs=thresh,
        exclude_border=True, indices=True)


#%%
def center_of_mass(x, y, z):
    """
    Compute the center_of_mass of a function in the x-y plane.

    Parameters
    ----------
    x : ndarray
        x-positions to consider
    y : ndarray
        y-positions to consider
    z : ndarray
        Weights at each point, x, y

    Returns
    -------
    output : ndarray, shape (2,)
        The x and y coordinates of the center of mass.
    """

    total = z.sum()
    return np.array([np.dot(x, z) / total, np.dot(y, z) / total])


#%%
def subpixel_locate(im, peak, selem, n_iters=20, fit_gauss=True, 
                    fit_gauss_bg=True, return_estimate_on_error=True, 
                    quiet=True):
    """
    Iterative fracshifts
    """

    # Make sure structuring element has more than four points for fit Gauss
    if selem.sum() <= 4 and fit_gauss:
        raise ValueError(
            'Can only have fit_guass = True with more than 4 pixels in selem.')

    # Get peak i and j for convenience
    i, j = peak

    # Get the i and j extent (radii) of the structuring element
    r = np.array(selem.shape) // 2
    r_i, r_j = r

    # Get indices of non-zero entries in structuring element for convenience
    ii, jj = np.nonzero(selem)

    # Define indices such that index zero is in center of selem
    i_pos = ii - r_i
    j_pos = jj - r_j

    # Width of structuring element
    w = 2 * r + 1

    # Make subimage that has selem, but with an extra pixel on all sides
    sub_im = im[i - r_i - 1:i + r_i + 2, j - r_j - 1:j + r_j + 2]

    # Compute fractional shift
    eps_i, eps_j = center_of_mass(i_pos, j_pos, sub_im[1:-1, 1:-1][ii, jj])

    # Now iterate on this process
    for _ in range(1, n_iters):
        if abs(eps_i) > 1 or abs(eps_j) > 1:
            if return_estimate_on_error:
                if not quiet:
                    print('Error in iterative fracshifting on iter %d' % _,
                          'Returning center of mass estimate.')
                return peak + center_of_mass(i_pos, j_pos,
                                             sub_im[1:-1, 1:-1][ii, jj])
            else:
                raise ValueError('Subpixel location is too big.')

        new_sub_im = scipy.ndimage.interpolation.shift(sub_im, (eps_i, eps_j),
                                                       order=1)

        eps_i, eps_j = center_of_mass(i_pos, j_pos,
                                      new_sub_im[1:-1, 1:-1][ii, jj])

    # Fit Gaussian to final subim
    if fit_gauss:
        print('We got here.')
        p_0 = _approx_gaussian_params(i_pos, j_pos,
                                      new_sub_im[1:-1, 1:-1][ii, jj])

        # Perform different regressions if we have background as well
        if fit_gauss_bg:
            p_0 = np.concatenate(((p_0[0],), (0.0,), p_0[1:]))
            fit_fun = _fit_gaussian_plus_background
        else:
            fit_fun = _fit_gaussian

        # Perform the Gaussian regression
        p, lsq_success = fit_fun(
            i_pos, j_pos, new_sub_im[1:-1, 1:-1][ii, jj], p_0)

        # Extract parameters that we need, also for validation
        eps_i, eps_j, sigma = p[-3:]

        # If we failed or don't validate, just return p_0
        if lsq_success:
            if not _bead_center_validate(eps_i, eps_j, sigma, w):
                if not quiet:
                    print('Warning: validation failed with values:')
                    print('    eps_i:', eps_i)
                    print('    eps_j:', eps_j)
                    print('    sigma:', sigma)
                    print('Setting center to center of mass estimate.')
                a, eps_i, eps_j, sigma = p_0
        else:
            if not quiet:
                print('Warning: Gaussian fitting in subpixel loc failed with:')
                print('    eps_i', eps_i)
                print('    eps_j:', eps_j)
                print('    sigma:', sigma)
                print('Setting center to center of mass estimate.')
            a, eps_i, eps_j, sigma = p_0

    return np.array([i + eps_i, j + eps_j])


#%%
def particle_centers(im, particle_size, blur=True, noise_size=1,
                     boxcar=True, fit_gauss=True, fit_gauss_bg=True, 
                     selem_width=3, selem_type='disk', n_iters=20, quiet=True):
    """
    Find the centers of all putative particles in an image

    im : ndarray
        The image containing particles.
    particle_size : float
        Diameter of particles in units of pixels.
    blur : bool
        If True, perform a Gaussian blur on each image.
    noise_size : float
        The characteristic length scale of noise in the images
        in units of pixels.  This is used to set the sigma value
        in the Gaussian blur.  Ignored if blur is False.
    boxcar : bool
        If True, perfrom a boxcar (mean) filter to compute background.
        The background is then subtracted.
    fit_gauss : bool
        True if least squares is used for subpixel localization.
        Otherwise, subpixel localization is done using moments.
    fit_gauss_bg : bool
        True if we include a background term in our Gaussian fit for
        subpixel localization.  Ignored if fit_gauss is False.
    selem_width : int
        The width of the structuring element used when doing
        subpixel localization.
    selem_type : int
        The type of structuring element to use in subpixel losalization,
        either 'disk' or 'square;.
    n_iters : int
        Numner of iterations of fracshifting to do in subpixel localization.
    """

    # Set particle size to the nearest odd integer (minimum of three)
    part_size = max(int((particle_size // 2) * 2 + 1), 3)

    # Set up boxcar width
    boxcar_width = 2 * part_size + 1

    # Set up structuring element for subpixel localization
    if selem_type == 'disk':
        selem = skimage.morphology.disk(selem_width // 2)
    elif selem_type != 'square':
        raise ValueError('Only disk and square selems allowed.')
    else:
        selem = skimage.morphology.square(selem_width)
 
   # Perform Gaussian blur and background subtraction
    im = preprocess(im, blur=blur, boxcar=boxcar, noise_size=noise_size,
                    boxcar_width=boxcar_width)

    # Find maxima to pixel resolution
    peaks_pixel = local_maxima_pixel(im, part_size)

    # Initialize array of centers
    centers = np.empty_like(peaks_pixel, dtype=np.float)

    # Get subpixel resolution
    for i, peak in enumerate(peaks_pixel):
        centers[i] = subpixel_locate(
            im, peak, selem, fit_gauss=fit_gauss, fit_gauss_bg=fit_gauss_bg,
            n_iters=n_iters, quiet=quiet)

    return centers


#%%
def centers_from_ims(im_list, particle_size, blur=True, noise_size=1,
                     boxcar=True, fit_gauss=True, fit_gauss_bg=True, 
                     selem_width=3, selem_type='disk', n_iters=20, 
                     quiet=False):
    """
    Compute centers of particles in a list of images.

    Parameters
    ----------
    im_list : tuple of strings
        tuple of file names of images.  The files must be readable
        by skimage.io.imread.  The index of the image in the list
        corresponds to the frame index of the movie.
    particle_size : float
        Diameter of particles in units of pixels.
    blur : bool
        If True, perform a Gaussian blur on each image.
    boxcar : bool
        If True, perfrom a boxcar (mean) filter to compute background.
        The background is then subtracted.
    noise_size : float
        The characteristic length scale of noise in the images
        in units of pixels.  This is used to set the sigma value
        in the Gaussian blur.  Ignored if blur is False.
    fit_gauss : bool
        True if least squares is used for subpixel localization.
        Otherwise, subpixel localization is done using moments.
    fit_gauss_bg : bool
        True if we include a background term in our Gaussian fit for
        subpixel localization.  Ignored if fit_gauss is False.
    selem_width : int
        The width of the structuring element used when doing
        subpixel localization.
    selem_type : int
        The type of structuring element to use in subpixel losalization,
        either 'disk' or 'square;.
    n_iters : int
        Numner of iterations of fracshifting to do in subpixel localization.
    quiet : bool
        True to supress progress to the screen

    Returns
    -------
    output : pandas DataFrame
        The indices of the DataFrame are the frame number.  Columns
        i and j contain the row and column coordinates of the pixel
        centers for the respective frame in units of pixels.
    """
    # Initialize DataFrame
    df = pd.DataFrame(columns=['i', 'j'])

    # Get total number of images to process
    n_images = len(im_list)

    for i, fname in enumerate(im_list):
        if not quiet and i % 100 == 0:
            print('Finding beads in image %d of %d....' % (i+1, n_images))

        im = _read_image(fname)

        centers = particle_centers(
            im, particle_size, blur=blur, noise_size=noise_size,
            fit_gauss=fit_gauss, fit_gauss_bg=fit_gauss_bg, 
            selem_width=selem_width, selem_type=selem_type, n_iters=n_iters, 
            quiet=quiet)

        index = np.ones(centers.shape[0]) * i
        df2 = pd.DataFrame(columns=['i', 'j'], index=index, data=centers)

        df = df.append(df2)

    return df


#%%
def _read_image(fname):
    """
    Read in an image.  If it has multiple channels, take first non-zero
    channel.

    Parameters
    ----------
    fname : string
        Name of the file containing the image data

    Returns
    -------
    output : ndarray
        The image as a NumPy array.
    """
    im = skimage.io.imread(fname)
    if len(im.shape) == 2:
        return im
    else:
        for i in range(im.shape[0]):
            if np.any(im[i, :, :] != 0):
                return im[i, :, :]
        return im[0, :, :]


#%% #########
def _sym_gaussian(x, y, p):
    """
    Compute a symmetric 2D Gaussian function,
    math::
        z(x,y) = a \exp\left[-\frac{((x - x_0)^2 + (y - y_0)^2)}
                                   {2 \sigma^2}\right]

    Parameters
    ----------
    x : ndarray
        Values of x-coordinates (independent variable) for data.
    y : ndarray
        Values of y-coordinates (independent variable) for data.
    p : array_like, shape (4,)
        The parameters (a, x_0, y_0, sigma)

    Returns
    -------
    output : ndarray, shape like x
        The symmetric 2D Gaussian function

    Returns a Gaussian function:
    a**2 * exp(-((x - x_0)**2 + (y - y_0)**2) / (2 * sigma**2))
    p = [a, x_0, y_0, sigma]
    """
    a, x_0, y_0, sigma = p
    return a**2 * np.exp(-((x - x_0)**2 + (y - y_0)**2) / (2.0 * sigma**2))


#%% #########
def _sym_gaussian_plus_background(x, y, p):
    """
    Compute a symmetric 2D Gaussian function,
    math::
        z(x,y) = b^2 + a^2 \exp\left[-\frac{((x - x_0)^2 + (y - y_0)^2)}
                                           {2 \sigma^2}\right]

    Parameters
    ----------
    x : ndarray
        Values of x-coordinates (independent variable) for data.
    y : ndarray
        Values of y-coordinates (independent variable) for data.
    p : array_like, shape (5,)
        The parameters (a, b, x_0, y_0, sigma)

    Returns
    -------
    output : ndarray, shape like x
        The symmetric 2D Gaussian plus background function
    """
    a, b, x_0, y_0, sigma = p
    return b**2 + a**2 * np.exp(-((x - x_0)**2 + (y - y_0)**2) \
                                    / (2.0 * sigma**2))


#%%
def _sym_gaussian_resids(p, x, y, z):
    """
    Residuals to be sent into leastsq

    Parameters
    ----------
    p : array_like, shape (4,)
        The parameters (a, x_0, y_0, sigma)
    x : ndarray
        Values of x-coordinates (independent variable) for data.
    y : ndarray
        Values of y-coordinates (independent variable) for data.
    z : ndarray
        Values of z-coordinates (dependent variable) for data.

    Returns
    -------
    output : array_like, shape like z
        The residuals between the computed symmetric Gaussian
        and the data.
    """
    return z - _sym_gaussian(x, y, p)


#%%
def _sym_gaussian_plus_background_resids(p, x, y, z):
    """
    Residuals to be sent into leastsq

    Parameters
    ----------
    p : array_like, shape (5,)
        The parameters (a, b, x_0, y_0, sigma)
    x : ndarray
        Values of x-coordinates (independent variable) for data.
    y : ndarray
        Values of y-coordinates (independent variable) for data.
    z : ndarray
        Values of z-coordinates (dependent variable) for data.

    Returns
    -------
    output : array_like, shape like z
        The residuals between the computed symmetric Gaussian
        and the data.
    """
    return z - _sym_gaussian_plus_background(x, y, p)


#%%
def _bead_center_validate(eps_i, eps_j, sigma, w):
    """
    Validate least sqaures result for subpixel bead center.

    Parameters
    ----------
    i : float
        Row position of particle center.
    j : float
        Column psotion of particle center.
    sigma : float
        Standard deviation of fit Gaussian.
    w : 2-tuple
        i and j width of structuring element used to determine center.

    Returns
    -------
    output : bool
        True if the curve fit parameters are reasonable, i.e.,
        small sigma, and center within the structuring element.
    """

    # Geometric mean of structuring element width
    w_gm = np.sqrt(w[0] * w[1])

    if (np.abs(eps_i) > w[0] / 2.0) or (np.abs(eps_j) > w[0] / 2.0) \
       or (sigma > w_gm):
        return False

    return True


#%%
def _approx_gaussian_params(x, y, z):
    """
    Estimates parameters of a symmetric Gaussian given by data x, y, z.
    I.e.,
    math::
        z(x,y) = a \exp\left[-\frac{((x - x_0)^2 + (y - y_0)^2)}
                                   {2 \sigma^2}\right]

    Parameters
    ----------
    x : ndarray
        Values of x-coordinates (independent variable) for data.
    y : ndarray
        Values of y-coordinates (independent variable) for data.
    z : ndarray
        Values of z-coordinates (dependent variable) for data.

    Returns
    -------
    a : float
        Parameter a
    x_0 : float
        Parameter x_0
    y_0 : float
        Parameter y_0
    sigma : float
        Parameter sigma

    Notes
    -----
    """
    a = z.max()

    # Compute moments
    total = z.sum()
    x_0 = np.dot(x, z) / total
    y_0 = np.dot(y, z) / total

    # Approximate sigmas
    sigma_x = np.dot(x**2, z) / total
    sigma_y = np.dot(y**2, z) / total
    sigma = np.sqrt(sigma_x * sigma_y)

    # Return estimate
    return np.array((a, x_0, y_0, sigma))


#%%
def _fit_gaussian(x, y, z, p_0):
    """
    Fits a symmetric Gaussian to data x, y, z.  I.e.,
    math::
        z(x,y) = a \exp\left[-\frac{((x - x_0)^2 + (y - y_0)^2)}
                                   {2 \sigma^2}\right]

    Parameters
    ----------
    x : ndarray
        Values of x-coordinates (independent variable) for data.
    y : ndarray
        Values of y-coordinates (independent variable) for data.
    z : ndarray
        Values of z-coordinates (dependent variable) for data.
    p_0 : array_like, shape (4,)
        Guesses for values of (a, x_0, y_0, sigma).  Usually these
        are generated by approx_gaussian_params.

    Returns
    -------
    params : ndarray
        Most probable (a, x_0, y_0, sigma) as an array
    lsq_success : bool
        True if least squares was successful.  If false, approximate
        values based on moments are returned in lsq_success.

    Notes
    -----
    .. Uses least squares.  This is not the fastest way.
    .. Returns the approximations of the parameters based on moments
       if least squares fails.
    """

    # Perform optimization using nonlinear least squares
    popt, junk_output, info_dict, mesg, ier = \
            scipy.optimize.leastsq(_sym_gaussian_resids, p_0, args=(x, y, z),
                                   full_output=True)

    # Check to make sure leastsq was successful.  If not, return centroid
    # estimate.
    if ier in (1, 2, 3, 4):
        return np.array((popt[0]**2, popt[1], popt[2], np.abs(popt[3]))), True
    else:
        return p_0, False
        
        
#%%
def _fit_gaussian_plus_background(x, y, z, p_0):
    """
    Fits a symmetric Gaussian to data x, y, z.  I.e.,
    math::
        z(x,y) = b + a \exp\left[-\frac{((x - x_0)^2 + (y - y_0)^2)}
                                       {2 \sigma^2}\right]

    Parameters
    ----------
    x : ndarray
        Values of x-coordinates (independent variable) for data.
    y : ndarray
        Values of y-coordinates (independent variable) for data.
    z : ndarray
        Values of z-coordinates (dependent variable) for data.
    p_0 : array_like, shape (5,)
        Guesses for values of (a, b, x_0, y_0, sigma).  Usually these
        are generated by approx_gaussian_params.

    Returns
    -------
    params : ndarray
        Most probable (a, x_0, y_0, sigma) as an array
    lsq_success : bool
        True if least squares was successful.  If false, approximate
        values based on moments are returned in lsq_success.

    Notes
    -----
    .. Uses least squares.  This is not the fastest way.
    .. Returns the approximations of the parameters based on moments
       if least squares fails.
    """

    # Perform optimization using nonlinear least squares
    popt, junk_output, info_dict, mesg, ier = \
            scipy.optimize.leastsq(_sym_gaussian_plus_background_resids, p_0, 
                                   args=(x, y, z), full_output=True)

    # Check to make sure leastsq was successful.  If not, return centroid
    # estimate.
    if ier in (1, 2, 3, 4):
        return np.array((popt[0]**2, popt[1]**2, popt[2], popt[3], 
                         np.abs(popt[4]))), True
    else:
        return p_0, False