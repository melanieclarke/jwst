import pytest
from gwcs.wcstools import grid_from_bounding_box
from numpy.testing import assert_allclose
from stcal.alignment import util
from stdatamodels.jwst import datamodels

from jwst.extract_1d import Extract1dStep
from jwst.regtest.st_fitsdiff import STFITSDiff as FITSDiff
from jwst.stpipe import Step

# Mark all tests in this module
pytestmark = [pytest.mark.bigdata]


@pytest.fixture(scope="module")
def run_pipeline(rtdata_module, resource_tracker):
    """Run the calwebb_spec2 pipeline on an ASN of nodded MIRI LRS
    fixedslit exposures."""
    rtdata = rtdata_module

    # Get the spec2 ASN and its members
    rtdata.get_asn("miri/lrs/jw01530-o005_20221202t204827_spec2_00001_asn.json")

    # Run the calwebb_spec2 pipeline; save results from intermediate steps
    args = [
        "calwebb_spec2",
        rtdata.input,
        "--steps.assign_wcs.save_results=true",
        "--save_bsub=true",
        "--steps.srctype.save_results=true",
        "--steps.flat_field.save_results=true",
        "--steps.pathloss.save_results=true",
        "--steps.pixel_replace.skip=false",
        "--steps.pixel_replace.save_results=true",
        "--steps.bkg_subtract.save_combined_background=true",
    ]
    with resource_tracker.track():
        Step.from_cmdline(args)


def test_log_tracked_resources_spec2(log_tracked_resources, run_pipeline):
    log_tracked_resources()


@pytest.mark.parametrize(
    "suffix",
    [
        "assign_wcs",
        "combinedbackground",
        "bsub",
        "srctype",
        "flat_field",
        "pathloss",
        "cal",
        "pixel_replace",
        "s2d",
        "x1d",
    ],
)
def test_miri_lrs_slit_spec2(run_pipeline, fitsdiff_default_kwargs, suffix, rtdata_module):
    """Regression test of the calwebb_spec2 pipeline on MIRI
    LRS fixedslit data using along-slit-nod pattern for
    background subtraction."""
    rtdata = rtdata_module
    output = f"jw01530005001_03103_00001_mirimage_{suffix}.fits"
    rtdata.output = output

    # Get the truth files
    rtdata.get_truth(f"truth/test_miri_lrs_slit_spec2/{output}")

    # Compare the results
    diff = FITSDiff(rtdata.output, rtdata.truth, **fitsdiff_default_kwargs)
    assert diff.identical, diff.report()


def test_miri_lrs_extract1d_from_cal(run_pipeline, rtdata_module, fitsdiff_default_kwargs):
    rtdata = rtdata_module
    rtdata.input = "jw01530005001_03103_00001_mirimage_cal.fits"
    Extract1dStep.call(rtdata.input, save_results=True)
    output = "jw01530005001_03103_00001_mirimage_extract1dstep.fits"
    rtdata.output = output
    rtdata.get_truth(f"truth/test_miri_lrs_slit_spec2/{output}")
    diff = FITSDiff(rtdata.output, rtdata.truth, **fitsdiff_default_kwargs)
    assert diff.identical, diff.report()


def test_miri_lrs_slit_wcs(run_pipeline, rtdata_module, fitsdiff_default_kwargs):
    rtdata = rtdata_module
    # get input assign_wcs and truth file
    output = "jw01530005001_03103_00001_mirimage_assign_wcs.fits"
    rtdata.output = output
    rtdata.get_truth(f"truth/test_miri_lrs_slit_spec2/{output}")

    # Compare the output and truth file
    with datamodels.open(output) as im, datamodels.open(rtdata.truth) as im_truth:
        x, y = grid_from_bounding_box(im.meta.wcs.bounding_box)
        ra, dec, lam = im.meta.wcs(x, y)
        ratruth, dectruth, lamtruth = im_truth.meta.wcs(x, y)
        assert_allclose(ra, ratruth)
        assert_allclose(dec, dectruth)
        assert_allclose(lam, lamtruth)

        # Test the inverse transform
        xtest, ytest = im.meta.wcs.backward_transform(ra, dec, lam)
        xtruth, ytruth = im_truth.meta.wcs.backward_transform(ratruth, dectruth, lamtruth)
        assert_allclose(xtest, xtruth)
        assert_allclose(ytest, ytruth)

        # Test the s_region. S_region is formed by footprint which contains
        # floats rather than a string. Test footprint
        sregion = im.meta.wcsinfo.s_region
        sregion_test = im_truth.meta.wcsinfo.s_region
        footprint = util.sregion_to_footprint(sregion)
        footprint_test = util.sregion_to_footprint(sregion_test)
        assert_allclose(footprint, footprint_test)
