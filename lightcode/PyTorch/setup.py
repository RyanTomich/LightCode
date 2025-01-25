from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="custom_kernel",
    ext_modules=[
        CppExtension("custom_kernel", ["kernel.cpp"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
