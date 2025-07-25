Description
===========

:Class: `jwst.source_catalog.SourceCatalogStep`
:Alias: source_catalog

This step creates a catalog of source photometry and morphologies.
Both aperture and isophotal (segment-based) photometry are calculated.
Source morphologies are based on 2D image moments within the source
segment.


Source Detection
----------------
Stars are detected in the input image with one of the following source
detection algorithms: ``photutils.detection.DAOStarFinder``,
``photutils.detection.IRAFStarFinder``, or ``photutils.segmentation.SourceFinder``
in conjunction with ``photutils.segmentation.SourceCatalog`` (default).

DAOStarFinder is an implementation of the `DAOFIND`_ algorithm
(`Stetson 1987, PASP 99, 191
<http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_).  It searches
images for local density maxima that have a peak amplitude greater
than a specified threshold (the threshold is applied to a convolved
image) and have a size and shape similar to a defined 2D Gaussian
kernel.  DAOFind also provides an estimate of the object's
roundness and sharpness, whose lower and upper bounds can be
specified.

IRAFStarFinder is a Python implementation of the IRAF star finding algorithm,
which also calculates the objects' centroids, roundness, and sharpness.
However, IRAFStarFinder uses image moments
instead of 1-D Gaussian fits to projected light distributions like
DAOStarFinder.

SourceFinder, the default option, implements an `image segmentation
<https://en.wikipedia.org/wiki/Image_segmentation>`_ algorithm, which is a
process of assigning a label to every pixel in an image such that
pixels with the same label are part of the same source.  The
segmentation procedure used is from `Photutils source extraction
<https://photutils.readthedocs.io/en/latest/segmentation.html>`_.
Detected sources must have a minimum number of connected pixels that
are each greater than a specified threshold value in an image.  The
threshold level is usually defined at some multiple of the background
standard deviation above the background.  The image can also be
filtered before thresholding to smooth the noise and maximize the
detectability of objects with a shape similar to the filter kernel.
Overlapping sources are detected as single sources.  Separating those
sources requires a deblending procedure, such as a multi-thresholding
technique used by `SExtractor
<https://www.astromatic.net/software/sextractor>`_.  Here we use the
`Photutils deblender
<https://photutils.readthedocs.io/en/latest/segmentation.html#source-deblending>`_,
which is an algorithm that deblends sources using a combination of
multi-thresholding and `watershed segmentation
<https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_.  In
order to deblend sources, they must be separated enough such that
there is a saddle between them.

.. warning::
    It has been shown (`STScI Technical Report JWST-STScI-008116, SM-12
    <https://www.stsci.edu/~goudfroo/NIRISSdoc/Centroid_Accuracies_Precisions_NIRISS_v2.pdf>`_)
    that for undersampled PSFs, e.g. for short-wavelength NIRISS
    imaging data, ``DAOStarFinder`` gives bad results no matter the input parameters
    due to its use of 1-D Gaussian fits.
    ``IRAFStarFinder`` or ``SourceFinder`` should be used instead.

.. note::
    If any other source detection algorithm other than ``SourceFinder`` is used,
    the output segmentation map will not be created, and the source catalog will
    be missing column values that are required for use as input to Level 2 spectral
    associations. Therefore if the direct image is to be used as part of a
    ``spec2`` association, ``SourceFinder`` should be used as the source
    detection algorithm.

.. _DAOFIND: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind


Source Photometry and Properties
--------------------------------
After detecting sources, we can measure their
photometry, centroids, and morphological properties.  The aperture
photometry is measured in three apertures, based on the input
encircled energy values.  The total aperture-corrected flux and
magnitudes are also calculated, based on the largest aperture.  Both
AB and Vega magnitudes are calculated.

The properties that are currently calculated for each source include
source centroids (both in pixel and sky coordinates), isophotal fluxes
(and errors), AB and Vega magnitudes (and errors), isophotal area,
semimajor and semiminor axis lengths, orientation of the major axis,
and sky coordinates at corners of the minimal bounding box enclosing
the source.

Photometric errors are calculated from the resampled total-error
array contained in the ``ERR`` (``model.err``) array. Note that this
total-error array includes source Poisson noise.

Source Position
---------------
The source centroid is computed as the center of mass of the unmasked pixels
within the source segment (see
`photutils SourceCatalog <https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html>`_
for details). As such, the centroid depends on the source morphology,
the parameters passed to the segmentation algorithm, and the local background
noise properties. This also makes the uncertainty in the centroid position
difficult to estimate, and a formal error estimate is not provided by the step.

Output Products
---------------

Source Catalog Table
^^^^^^^^^^^^^^^^^^^^
The output source catalog table is saved in `ECSV format
<https://docs.astropy.org/en/stable/io/ascii/ecsv.html>`_.

The table contains a row for each source, with the following default
columns (assuming the default encircled energies of 30, 50, and 70):

+------------------------+----------------------------------------------------+
| Column                 | Description                                        |
+========================+====================================================+
| label                  | Unique source identification label number          |
+------------------------+----------------------------------------------------+
| xcentroid              | X pixel value of the source centroid (0 indexed)   |
+------------------------+----------------------------------------------------+
| ycentroid              | Y pixel value of the source centroid (0 indexed)   |
+------------------------+----------------------------------------------------+
| sky_centroid           | Sky coordinate of the source centroid              |
+------------------------+----------------------------------------------------+
| aper_bkg_flux          | The local background value calculated as the       |
|                        | sigma-clipped median value in the background       |
|                        | annulus aperture                                   |
+------------------------+----------------------------------------------------+
| aper_bkg_flux_err      | The standard error of the sigma-clipped median     |
|                        | background value                                   |
+------------------------+----------------------------------------------------+
| aper30_flux            | Flux within the 30% encircled energy circular      |
|                        | aperture                                           |
+------------------------+----------------------------------------------------+
| aper30_flux_err        | Flux error within the 30% encircled energy         |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper50_flux            | Flux within the 50% encircled energy circular      |
|                        | aperture                                           |
+------------------------+----------------------------------------------------+
| aper50_flux_err        | Flux error within the 50% encircled energy         |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper70_flux            | Flux within the 70% encircled energy circular      |
|                        | aperture                                           |
+------------------------+----------------------------------------------------+
| aper70_flux_err        | Flux error within the 70% encircled energy         |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper_total_flux        | Total aperture-corrected flux based on the 70%     |
|                        | encircled energy circular aperture; should be used |
|                        | only for unresolved sources                        |
+------------------------+----------------------------------------------------+
| aper_total_flux_err    | Total aperture-corrected flux error based on the   |
|                        | 70% encircled energy circular aperture; should be  |
|                        | used only for unresolved sources                   |
+------------------------+----------------------------------------------------+
| aper30_abmag           | AB magnitude within the 30% encircled energy       |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper30_abmag_err       | AB magnitude error within the 30% encircled energy |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper50_abmag           | AB magnitude within the 50% encircled energy       |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper50_abmag_err       | AB magnitude error within the 50% encircled energy |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper70_abmag           | AB magnitude within the 70% encircled energy       |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper70_abmag_err       | AB magnitude error within the 70% encircled energy |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper_total_abmag       | Total aperture-corrected AB magnitude based on the |
|                        | 70% encircled energy circular aperture; should be  |
|                        | used only for unresolved sources                   |
+------------------------+----------------------------------------------------+
| aper_total_abmag_err   | Total aperture-corrected AB magnitude error based  |
|                        | on the 70% encircled energy circular aperture;     |
|                        | should be used only for unresolved sources         |
+------------------------+----------------------------------------------------+
| aper30_vegamag         | Vega magnitude within the 30% encircled energy     |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper30_vegamag_err     | Vega magnitude error within the 30% encircled      |
|                        | energy circular aperture                           |
+------------------------+----------------------------------------------------+
| aper50_vegamag         | Vega magnitude within the 50% encircled energy     |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper50_vegamag_err     | Vega magnitude error within the 50% encircled      |
|                        | energy circular aperture                           |
+------------------------+----------------------------------------------------+
| aper70_vegamag         | Vega magnitude within the 70% encircled energy     |
|                        | circular aperture                                  |
+------------------------+----------------------------------------------------+
| aper70_vegamag_err     | Vega magnitude error within the 70% encircled      |
|                        | energy circular aperture                           |
+------------------------+----------------------------------------------------+
| aper_total_vegamag     | Total aperture-corrected Vega magnitude based on   |
|                        | the 70% encircled energy circular aperture;        |
|                        | should be used only for unresolved sources         |
+------------------------+----------------------------------------------------+
| aper_total_vegamag_err | Total aperture-corrected Vega magnitude error      |
|                        | based on the 70% encircled energy circular         |
|                        | aperture; should be used only for unresolved       |
|                        | sources                                            |
+------------------------+----------------------------------------------------+
| CI_50_30               | Concentration index calculated as (aper50_flux /   |
|                        | aper30_flux)                                       |
+------------------------+----------------------------------------------------+
| CI_70_50               | Concentration index calculated as (aper70_flux /   |
|                        | aper50_flux)                                       |
+------------------------+----------------------------------------------------+
| CI_70_30               | Concentration index calculated as (aper70_flux /   |
|                        | aper30_flux)                                       |
+------------------------+----------------------------------------------------+
| is_extended            | Flag indicating whether the source is extended     |
+------------------------+----------------------------------------------------+
| sharpness              | The DAOFind source sharpness statistic             |
+------------------------+----------------------------------------------------+
| roundness              | The DAOFind source roundness statistic             |
+------------------------+----------------------------------------------------+
| nn_label               | The label number of the nearest neighbor           |
+------------------------+----------------------------------------------------+
| nn_dist                | The distance in pixels to the nearest neighbor     |
+------------------------+----------------------------------------------------+
| isophotal_flux         | Isophotal flux                                     |
+------------------------+----------------------------------------------------+
| isophotal_flux_err     | Isophotal flux error                               |
+------------------------+----------------------------------------------------+
| isophotal_abmag        | Isophotal AB magnitude                             |
+------------------------+----------------------------------------------------+
| isophotal_abmag_err    | Isophotal AB magnitude error                       |
+------------------------+----------------------------------------------------+
| isophotal_vegamag      | Isophotal Vega magnitude                           |
+------------------------+----------------------------------------------------+
| isophotal_vegamag_err  | Isophotal Vega magnitude error                     |
+------------------------+----------------------------------------------------+
| isophotal_area         | Isophotal area                                     |
+------------------------+----------------------------------------------------+
| semimajor_sigma        | 1-sigma standard deviation along the semimajor     |
|                        | axis of the 2D Gaussian function that has the same |
|                        | second-order central moments as the source         |
+------------------------+----------------------------------------------------+
| semiminor_sigma        | 1-sigma standard deviation along the semiminor     |
|                        | axis of the 2D Gaussian function that has the same |
|                        | second-order central moments as the source         |
+------------------------+----------------------------------------------------+
| ellipticity            | 1 minus the ratio of the 1-sigma lengths of the    |
|                        | semimajor and semiminor axes                       |
+------------------------+----------------------------------------------------+
| orientation            | The angle (degrees) between the positive X axis    |
|                        | and the major axis (increases counter-clockwise)   |
+------------------------+----------------------------------------------------+
| sky_orientation        | The position angle (degrees) from North of the     |
|                        | major axis                                         |
+------------------------+----------------------------------------------------+
| sky_bbox_ll            | Sky coordinate of the lower-left vertex of the     |
|                        | minimal bounding box of the source                 |
+------------------------+----------------------------------------------------+
| sky_bbox_ul            | Sky coordinate of the upper-left vertex of the     |
|                        | minimal bounding box of the source                 |
+------------------------+----------------------------------------------------+
| sky_bbox_lr            | Sky coordinate of the lower-right vertex of the    |
|                        | minimal bounding box of the source                 |
+------------------------+----------------------------------------------------+
| sky_bbox_ur            | Sky coordinate of the upper-right vertex of the    |
|                        | minimal bounding box of the source                 |
+------------------------+----------------------------------------------------+

Note that pixel coordinates are 0 indexed, matching the Python 0-based
indexing. That means pixel coordinate ``0`` is the center of the first
pixel.


Segmentation Map
^^^^^^^^^^^^^^^^
The segmentation map computed during the source finding process is saved
to a single 2D image extension in a FITS file. Each image pixel contains an
integer value corresponding to a source label number in the source catalog
product. Pixels that don't belong to any source have a value of zero.
