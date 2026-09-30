from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

extra_compile_args = {
    "nvcc": [
        "-allow-unsupported-compiler",
        "-lineinfo",
        "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "glm"),
    ]
}


setup(
    name="diff_gaussian_rasterization_light",
    packages=['diff_gaussian_rasterization_light'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_light._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"
            ],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
