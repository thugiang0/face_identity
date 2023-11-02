import setuptools 


try:
    # pip >= 20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements

install_reqs = parse_requirements("requirements.txt", session=False)
try:
    reqs = [str(ir.req) for ir in install_reqs]
except:
    reqs = [str(ir.requirement) for ir in install_reqs]
  
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
    python_requires='>=3.8', 
    install_requires=reqs,
    include_package_data=True,
) 