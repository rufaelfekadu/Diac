from pathlib import Path
from setuptools import setup, find_packages
import io
import re


HERE = Path(__file__).parent


def read_readme():
    readme = HERE / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


def parse_requirements(path):
    """Parse a requirements.txt-ish file into a list suitable for
    install_requires. This tolerant parser skips fences, headings and
    comments so it works with the project's current requirements file.
    """
    p = Path(path)
    if not p.exists():
        return []

    lines = p.read_text(encoding="utf-8").splitlines()
    reqs = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # skip fenced code blocks or file headers sometimes present
        if ln.startswith("```") or ln.lower().startswith("pip-requirements"):
            continue
        if ln.startswith("#"):
            continue
        # basic sanity: skip non-dependency lines
        if re.match(r"^[A-Za-z0-9_\-\.\[\]]+\s*(?:[<>=!~]=?).*$", ln) or re.search(
            r"\d", ln
        ):
            reqs.append(ln)
        else:
            # fall back: include the line if it looks like a package
            if "/" not in ln and "git+" not in ln:
                reqs.append(ln)
    return reqs


setup(
    name="diac",
    version="0.1.0",
    description="Arabic diacritization using text and audio.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    # Use src/ as the package root so code inside `src/` is installed as
    # top-level packages. This follows the common "src layout" for Python
    # projects. If you prefer a single top-level package name, move the
    # package directory under `src/<pkg_name>/` and adjust package metadata.
    packages=find_packages(
        where="src", exclude=("data", "outputs", "results", "tests")
    ),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        # include pickles, configs and other non-code assets commonly used by the project
        "": ["*.yml", "*.yaml", "*.pickle", "*.pt", "*.txt"]
    },
    python_requires=">=3.8",
    install_requires=parse_requirements(HERE / "requirements.txt"),
    license="MIT",
    author="",
    author_email="",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
