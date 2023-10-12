import setuptools 
  
with open("README.md", "r") as fh: 
    description = fh.read() 
  
setuptools.setup( 
    name="face_recognition", 
    version="0.0.1", 
    author="giangtt", 
    # packages=["face_recognition"],
    packages=setuptools.find_packages(), 
    description="A sample test package", 
    long_description=description, 
    long_description_content_type="text/markdown", 
    license='MIT', 
    python_requires='>=3.7', 
    install_requires=[],
    include_package_data=True,
) 