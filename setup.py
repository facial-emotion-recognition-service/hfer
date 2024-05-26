from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hfer",
    # version="0.0.1",
    description="Human Facial Emotion Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # license="MIT",
    # author="",
    # author_email="",
    # url="",
    install_requires=requirements,
    extras_require={"dev": ["ipdb"]},
    python_requires=">=3.10",
    packages=find_packages(),
    # scripts=['hfer/scripts/test_script.py'],
    # test_suite="tests",
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    # zip_safe=False,
)
