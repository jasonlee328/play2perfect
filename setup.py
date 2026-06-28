from setuptools import find_packages, setup

# Minimal dependencies; the full set (incl. Isaac Sim / Isaac Lab / torch) is documented
# in docs/isaacsim_installation.md and installed separately. See pyproject.toml.
INSTALL_REQUIRES = [
    "gym==0.23.1",
    "omegaconf",
    "hydra-core>=1.2",
    "termcolor",
    "jinja2",
    "viser",
    "tyro",
    "requests",
    "tqdm",
    "huggingface_hub",
]

setup(
    name="play2perfect",
    version="0.1.0",
    author="Kushal Kedia",
    author_email="kk837@cornell.edu",
    url="https://github.com/kushal2000/play2perfect",
    description="Official implementation of Play2Perfect: a 2-stage RL pipeline for contact-rich assembly",
    keywords=["robotics", "rl", "assembly", "manipulation"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(include=["isaacsimenvs*", "evaluation*", "deployment*"]),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.11"],
    zip_safe=False,
)
