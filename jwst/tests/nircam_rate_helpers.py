"""Mock NIRCam data in rate format."""

import numpy as np
from astropy.io import fits
from gwcs import wcs
from stdatamodels.jwst import datamodels

from jwst.assign_wcs import nircam
from jwst.tests.wcs_helpers import get_reference_files

__all_ = ["nircam_image_model", "nircam_tsgrism_rateints_model", "nircam_wfss_rate_model"]

# Default wcs information
DEFAULT_WCS_KW = {
    "wcsaxes": 2,
    "ra_ref": 53.1423683802,
    "dec_ref": -27.8171119969,
    "v2_ref": 86.103458,
    "v3_ref": -493.227512,
    "roll_ref": 45.04234459270135,
    "crpix1": 1024.5,
    "crpix2": 1024.5,
    "crval1": 53.1423683802,
    "crval2": -27.8171119969,
    "cdelt1": 1.74460027777777e-05,
    "cdelt2": 1.75306861111111e-05,
    "ctype1": "RA---TAN",
    "ctype2": "DEC--TAN",
    "pc1_1": -1,
    "pc1_2": 0,
    "pc2_1": 0,
    "pc2_2": 1,
    "cunit1": "deg",
    "cunit2": "deg",
}


# Example keyword values from jw01366002001_04103_00001
TSO_WCS_KW = {
    "wcsaxes": 2,
    "ra_ref": 217.3262354867486,
    "dec_ref": -3.444360390029802,
    "v2_ref": 73.106857,
    "v3_ref": -551.643196,
    "v3i_yang": 0.24336066,
    "vparity": -1,
    "roll_ref": 110.12344444842617,
    "xref_sci": 1581.0,
    "yref_sci": 35.0,
    "cdelt1": 1.76686111111111e-05,
    "cdelt2": 1.78527777777777e-05,
    "ctype1": "RA---TAN",
    "ctype2": "DEC--TAN",
    "pc1_1": -1,
    "pc1_2": 0,
    "pc2_1": 0,
    "pc2_2": 1,
    "cunit1": "deg",
    "cunit2": "deg",
}


def _nircam_rate_hdul(
    detector="NRCALONG",
    channel="LONG",
    module="A",
    filtername="F444W",
    exptype="NRC_IMAGE",
    pupil="GRISMR",
    subarray="FULL",
    wcskeys=None,
):
    if wcskeys is None:
        wcskeys = DEFAULT_WCS_KW

    hdul = fits.HDUList()
    phdu = fits.PrimaryHDU()
    phdu.header["TELESCOP"] = "JWST"
    phdu.header["FILENAME"] = "test+" + filtername
    phdu.header["INSTRUME"] = "NIRCAM"
    phdu.header["CHANNEL"] = channel
    phdu.header["DETECTOR"] = detector
    phdu.header["FILTER"] = filtername
    phdu.header["PUPIL"] = pupil
    phdu.header["MODULE"] = module
    phdu.header["TIME-OBS"] = "8:59:37"
    phdu.header["DATE-OBS"] = "2023-01-01"
    phdu.header["EXP_TYPE"] = exptype
    phdu.header["SUBARRAY"] = subarray
    scihdu = fits.ImageHDU()
    scihdu.header["EXTNAME"] = "SCI"
    scihdu.header.update(wcskeys)
    hdul.append(phdu)
    hdul.append(scihdu)
    return hdul


def nircam_image_rate_model(with_wcs=True):
    """
    Create a mock NIRCam image rate model.

    The data array is zero-filled with shape (10, 10).

    Parameters
    ----------
    with_wcs : bool, optional
        If True, assign a WCS to the output model.

    Returns
    -------
    model : `~stdatamodels.jwst.datamodels.ImageModel`
        The NIRCam image datamodel.
    """
    model = datamodels.ImageModel(_nircam_rate_hdul())
    model.data = np.zeros((10, 10))

    if with_wcs:
        ref = get_reference_files(model)
        pipeline = nircam.create_pipeline(model, ref)
        model.meta.wcs = wcs.WCS(pipeline)

    return model


def nircam_tsgrism_rateints_model(filtername="F322W2", with_wcs=True):
    """
    Create a mock NIRCam TSGRISM rateints model.

    The data array is zero-filled with shape (10, 10, 10).

    Parameters
    ----------
    with_wcs : bool, optional
        If True, assign a WCS to the output model.

    Returns
    -------
    model : `~stdatamodels.jwst.datamodels.CubeModel`
        The NIRCam image datamodel.
    """
    hdul = _nircam_rate_hdul(
        exptype="NRC_TSGRISM",
        pupil="GRISMR",
        filtername=filtername,
        detector="NRCALONG",
        subarray="SUBGRISM256",
        wcskeys=TSO_WCS_KW,
    )
    model = datamodels.CubeModel(hdul)
    model.data = np.zeros((10, 10, 10))
    model.meta.dither.x_offset = 0.0
    model.meta.dither.y_offset = 1.4485  # example from jw01366002001_04103_00001

    if with_wcs:
        ref = get_reference_files(model)
        pipeline = nircam.create_pipeline(model, ref)
        model.meta.wcs = wcs.WCS(pipeline)

    return model


def nircam_wfss_rate_model(pupil="GRISMR", with_wcs=True):
    """
    Create a mock NIRCam WFSS rateints model.

    The data array is zero-filled with shape (10, 10).

    Parameters
    ----------
    with_wcs : bool, optional
        If True, assign a WCS to the output model.

    Returns
    -------
    model : `~stdatamodels.jwst.datamodels.CubeModel`
        The NIRCam image datamodel.
    """
    hdul = _nircam_rate_hdul(exptype="NRC_WFSS", filtername="F444W", pupil=pupil)
    model = datamodels.ImageModel(hdul)
    model.data = np.zeros((10, 10))

    if with_wcs:
        ref = get_reference_files(model)
        pipeline = nircam.create_pipeline(model, ref)
        model.meta.wcs = wcs.WCS(pipeline)

    return model
