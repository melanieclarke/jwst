import os
import os.path as op
import pprint
import shutil
import sys
from difflib import unified_diff
from glob import glob as _sys_glob
from pathlib import Path

import asdf
import requests
from ci_watson.artifactory_helpers import (
    BigdataError,
    check_url,
    get_bigdata,
    get_bigdata_root,
)

from jwst.associations import AssociationNotValidError, load_asn
from jwst.lib.file_utils import pushdir
from jwst.lib.suffix import replace_suffix
from jwst.pipeline.collect_pipeline_cfgs import collect_pipeline_cfgs
from jwst.regtest.st_fitsdiff import STFITSDiff as FITSDiff
from jwst.stpipe import Step

# Define location of default Artifactory API key
ARTIFACTORY_API_KEY_FILE = "/eng/ssb2/keys/svc_rodata.key"


class RegtestData:
    """Define data paths on Artifactory and data retrieval methods."""

    def __init__(
        self,
        env="dev",
        inputs_root="jwst-pipeline",
        results_root="jwst-pipeline-results",
        docopy=True,
        input=None,
        input_remote=None,
        output=None,
        truth=None,
        truth_remote=None,
        remote_results_path=None,
        test_name=None,
        traceback=None,
        okify_op="file_copy",
        **kwargs,
    ):
        self.env = env
        self._inputs_root = inputs_root
        self._results_root = results_root
        self._bigdata_root = get_bigdata_root()

        self.docopy = docopy

        # Initialize @property attributes
        self.input = input
        self.input_remote = input_remote
        self.output = output
        self.truth = truth
        self.truth_remote = truth_remote

        # No @properties for the following attributes
        self.remote_results_path = remote_results_path
        self.test_name = test_name
        self.traceback = traceback
        self.okify_op = okify_op

        # Initialize non-initialized attributes
        self.asn = None

    def __repr__(self):
        return pprint.pformat(
            {
                "input": self.input,
                "output": self.output,
                "truth": self.truth,
                "input_remote": self.input_remote,
                "truth_remote": self.truth_remote,
                "remote_results_path": self.remote_results_path,
                "test_name": self.test_name,
                "traceback": self.traceback,
                "okify_op": self.okify_op,
            },
            indent=1,
        )

    @property
    def input_remote(self):  # numpydoc ignore=RT01
        """Return the remote input path."""
        if self._input_remote is not None:
            return os.path.join(*self._input_remote)
        else:
            return None

    @input_remote.setter
    def input_remote(self, value):
        if value:
            self._input_remote = value.split(os.sep)
        else:
            self._input_remote = value

    @property
    def truth_remote(self):  # numpydoc ignore=RT01
        """Return the remote truth path."""
        if self._truth_remote is not None:
            return os.path.join(*self._truth_remote)
        else:
            return None

    @truth_remote.setter
    def truth_remote(self, value):
        if value:
            self._truth_remote = value.split(os.sep)
        else:
            self._truth_remote = value

    @property
    def input(self):  # numpydoc ignore=RT01
        """Return the local input path."""
        return self._input

    @input.setter
    def input(self, value):
        if value:
            self._input = os.path.abspath(value)
        else:
            self._input = value

    @property
    def truth(self):  # numpydoc ignore=RT01
        """Return the local truth path."""
        return self._truth

    @truth.setter
    def truth(self, value):
        if value:
            self._truth = os.path.abspath(value)
        else:
            self._truth = value

    @property
    def output(self):  # numpydoc ignore=RT01
        """Return the local output path."""
        return self._output

    @output.setter
    def output(self, value):
        if value:
            self._output = os.path.abspath(value)
        else:
            self._output = value

    @property
    def bigdata_root(self):  # numpydoc ignore=RT01
        """Return the root of the bigdata server."""
        return self._bigdata_root

    # The methods
    def get_data(self, path=None, docopy=None):
        """
        Copy data from Artifactory remote resource to the CWD.

        Updates self.input and self.input_remote upon completion

        Parameters
        ----------
        path : str, optional
            The path to the input file that will be copied.
        docopy : bool, optional
            Switch to control whether or not to copy a file
            into the test output directory when running the test.
            If you wish to open the file directly from remote
            location or just to set path to source, set this to `False`.
            Default: `True`

        Returns
        -------
        input : str
            The full local path to the input file.
        """
        if path is None:
            path = self.input_remote
        else:
            self.input_remote = path
        if docopy is None:
            docopy = self.docopy
        self.input = get_bigdata(self._inputs_root, self.env, path, docopy=docopy)
        self.input_remote = os.path.join(self._inputs_root, self.env, path)

        return self.input

    def data_glob(self, path=None, glob="*"):
        """
        Get a list of files from either a local path or URL.

        Parameters
        ----------
        path : str, optional
            The path to the input file that will be copied.
            glob : str, optional
            The glob pattern to use to search for files.
            Default: `*`

        Returns
        -------
        file_paths : [str[, ...]]
            Full file paths that match the glob criterion
        """
        if path is None:
            path = self.input_remote
        else:
            self.input_remote = path

        # Get full path and proceed depending on whether
        # is a local path or URL.
        root = self.bigdata_root
        if op.exists(root):
            root_path = op.join(root, self._inputs_root, self.env)
            root_len = len(root_path) + 1
            path = op.join(root_path, path)
            file_paths = _data_glob_local(path, glob)
        elif check_url(root):
            root_len = len(self.env) + 1
            file_paths = _data_glob_url(self._inputs_root, self.env, path, glob, root=root)
        else:
            raise BigdataError(f"Path cannot be found: {path}")

        # Remove the root from the paths
        file_paths = [file_path[root_len:] for file_path in file_paths]
        return file_paths

    def get_truth(self, path=None, docopy=None):
        """
        Copy truth data from Artifactory remote resource to the CWD/truth.

        Updates self.truth and self.truth_remote on completion

        Parameters
        ----------
        path : str, optional
            The path to the truth file that will be copied.
        docopy : bool, optional
            Switch to control whether or not to copy a file
            into the test output directory when running the test.
            If you wish to open the file directly from remote
            location or just to set path to source, set this to `False`.
            Default: `True`

        Returns
        -------
        str
            The local path to the truth file.
        """
        if path is None:
            path = self.truth_remote
        else:
            self.truth_remote = path
        if docopy is None:
            docopy = self.docopy
        os.makedirs("truth", exist_ok=True)
        with pushdir("truth"):
            self.truth = get_bigdata(self._inputs_root, self.env, path, docopy=docopy)
            self.truth_remote = os.path.join(self._inputs_root, self.env, path)

        return self.truth

    def get_asn(self, path=None, docopy=True, get_members=True):
        """
        Copy association and association members from Artifactory remote resource to the CWD/truth.

        Updates self.input and self.input_remote upon completion

        Parameters
        ----------
        path : str
            The remote path

        docopy : bool
            Switch to control whether or not to copy a file
            into the test output directory when running the test.
            If you wish to open the file directly from remote
            location or just to set path to source, set this to `False`.
            Default: `True`

        get_members : bool
            If an association is the input, retrieve the members.
            Otherwise, do not.
        """
        if path is None:
            path = self.input_remote
        else:
            self.input_remote = path
        if docopy is None:
            docopy = self.docopy

        # Get the association JSON file
        self.input = get_bigdata(self._inputs_root, self.env, path, docopy=docopy)
        with open(self.input) as fp:
            asn = load_asn(fp)
            self.asn = asn

        # Get each member in the association as well
        if get_members:
            for product in asn["products"]:
                for member in product["members"]:
                    fullpath = os.path.join(os.path.dirname(self.input_remote), member["expname"])
                    get_bigdata(self._inputs_root, self.env, fullpath, docopy=self.docopy)

    def to_asdf(self, path):
        """Write the RegtestData object to an ASDF file."""
        tree = eval(str(self))  # noqa: S307
        af = asdf.AsdfFile(tree=tree)
        af.write_to(path)

    @classmethod
    def open(cls, filename):
        """
        Read a RegtestData object from an ASDF file.

        Parameters
        ----------
        filename : str
            The path to the ASDF file

        Returns
        -------
        `jwst.regtest.regtestdata.RegtestData`
            The RegtestData object
        """
        with asdf.open(filename) as af:
            return cls(**af.tree)


def run_step_from_dict(rtdata, **step_params):
    """
    Run Step on regression test data from a dictionary of input parameters.

    Parameters
    ----------
    rtdata : RegtestData
        The artifactory instance

    **step_params : dict
        The parameters defining what step to run with what input

    Returns
    -------
    rtdata : RegtestData
        Updated `RegtestData` object with inputs set.

    Notes
    -----
    `step_params` looks like this:
    {
        'input_path': str or None  # The input file path, relative to artifactory
        'step': str                # The step to run, either a class or a config file
        'args': list,              # The arguments passed to `Step.from_cmdline`
    }
    """
    # Get the data. If `step_params['input_path]` is not
    # specified, the presumption is that `rtdata.input` has
    # already been retrieved.
    input_path = step_params.get("input_path", None)
    if input_path:
        try:
            rtdata.get_asn(input_path)
        except AssociationNotValidError:
            rtdata.get_data(input_path)

    # Figure out whether we have a config or class
    step = step_params["step"]
    if step.endswith((".asdf", ".cfg")):
        step = os.path.join("config", step)

    # Run the step
    collect_pipeline_cfgs("config")
    full_args = [step, rtdata.input]
    full_args.extend(step_params["args"])

    Step.from_cmdline(full_args)

    return rtdata


def run_step_from_dict_mock(rtdata, source, **step_params):
    """
    Pretend to run Steps with given parameter but just copy data.

    For long running steps where the result already exists, just
    copy the data from source

    Parameters
    ----------
    rtdata : RegtestData
        The artifactory instance

    source : Path-like folder
        The folder to copy from. All regular files are copied.

    **step_params : dict
        The parameters defining what step to run with what input

    Returns
    -------
    rtdata : RegtestData
        Updated `RegtestData` object with inputs set.

    Notes
    -----
    `step_params` looks like this:
    {
        'input_path': str or None  # The input file path, relative to artifactory
        'step': str                # The step to run, either a class or a config file
        'args': list,              # The arguments passed to `Step.from_cmdline`
    }
    """
    # Get the data. If `step_params['input_path]` is not
    # specified, the presumption is that `rtdata.input` has
    # already been retrieved.
    input_path = step_params.get("input_path", None)
    if input_path:
        try:
            rtdata.get_asn(input_path)
        except AssociationNotValidError:
            rtdata.get_data(input_path)

    # Copy the data
    for file_name in os.listdir(source):
        file_path = os.path.join(source, file_name)
        if os.path.isfile(file_path):
            shutil.copy(file_path, ".")

    return rtdata


def is_like_truth(rtdata, fitsdiff_default_kwargs, output, truth_path, is_suffix=True):
    """
    Compare step outputs with truth.

    Parameters
    ----------
    rtdata : RegtestData
        The artifactory object from the step run.

    fitsdiff_default_kwargs : dict
        The `fitsdiff` keyword arguments

    output : str
        The suffix or full file name to check on.

    truth_path : str
        Location of the truth files.

    is_suffix : bool
        Interpret `output` as just a suffix on the expected output root.
        Otherwise, assume it is a full file name
    """
    __tracebackhide__ = True
    # If given only a suffix, get the root to change the suffix of.
    # If the input was an association, the output should be the name of the product
    # Otherwise, output is based on input.
    if is_suffix:
        suffix = output
        if rtdata.asn:
            output = rtdata.asn["products"][0]["name"]
        else:
            output = os.path.splitext(os.path.basename(rtdata.input))[0]
        output = replace_suffix(output, suffix) + ".fits"
    rtdata.output = output

    rtdata.get_truth(os.path.join(truth_path, output))

    diff = FITSDiff(rtdata.output, rtdata.truth, **fitsdiff_default_kwargs)
    assert diff.identical, diff.report()


def text_diff(from_path, to_path):
    """
    Diff two text files.

    Parameters
    ----------
    from_path : str
        File to diff from.

    to_path : str
        File to diff to.  The truth.

    Returns
    -------
    diffs : [str[,...]]
        A generator of a list of strings that are the differences.
        The output from `difflib.unified_diff`
    """
    __tracebackhide__ = True
    with open(from_path) as fh:
        from_lines = fh.readlines()
    with open(to_path) as fh:
        to_lines = fh.readlines()

    # convert path objects to strings because difflib requires strings
    diffs = unified_diff(from_lines, to_lines, str(from_path), str(to_path))

    diff = list(diffs)
    if len(diff) > 0:
        diff.insert(0, "\n")
        raise AssertionError("".join(diff))
    else:
        return True


def _data_glob_local(*glob_parts):
    """
    Perform a glob on the local path.

    Parameters
    ----------
    *glob_parts : (path-like,[...])
        List of components that will be built into a single path

    Returns
    -------
    file_paths : [str[, ...]]
        Full file paths that match the glob criterion
    """
    full_glob = Path().joinpath(*glob_parts)
    return _sys_glob(str(full_glob))


def _data_glob_url(*url_parts, root=None):
    """
    Perform a glob on a URL path.

    Parameters
    ----------
    *url_parts : (str[,...])
        List of components that will be used to create a URL path

    root : str
        The root server path to the Artifactory server.
        Normally retrieved from `get_bigdata_root`.

    Returns
    -------
    url_paths : [str[, ...]]
        Full URLS that match the glob criterion
    """
    # Fix root root-ed-ness
    if root.endswith("/"):
        root = root[:-1]

    # Access
    try:
        envkey = os.environ["API_KEY_FILE"]
    except KeyError:
        envkey = ARTIFACTORY_API_KEY_FILE

    try:
        with open(envkey) as fp:
            headers = {"X-JFrog-Art-Api": fp.readline().strip()}
    except (PermissionError, FileNotFoundError):
        print(  # noqa: T201
            "Warning: Anonymous Artifactory search requests are limited to "  # noqa: T201
            "1000 results. Use an API key and define API_KEY_FILE environment "
            "variable to get full search results.",
            file=sys.stderr,
        )
        headers = None

    search_url = "/".join([root, "api/search/pattern"])

    # Join and re-split the url so that every component is identified.
    url = "/".join([root] + list(url_parts))
    all_parts = url.split("/")

    # Pick out "jwst-pipeline", the repo name
    repo = all_parts[4]

    # Format the pattern
    pattern = repo + ":" + "/".join(all_parts[5:])

    # Make the query
    params = {"pattern": pattern}
    with requests.get(search_url, params=params, headers=headers, timeout=60) as r:
        url_paths = r.json()["files"]

    return url_paths
