import os
import re
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

from compile_additional_binaries import build_cmake_project


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    def run(self):
        if platform.system() != "Windows":
            raise RuntimeError('This library currently only supports Windows')

        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < '3.1.0':
            raise RuntimeError("CMake >= 3.1.0 is required")

        for ext in self.extensions:
            build_cmake_project(
                '.',
                self.build_temp,
                os.path.dirname(self.get_ext_fullpath(ext.name)),
                None,
                'pyrlhook',
                False,
                self.debug
            )


if __name__ == '__main__':

    name = 'pyrlhook'
    # https://dzone.com/articles/executable-package-pip-install
    setup(
        name=name,
        version='0.0.1',
        author='Julius Hansjakob',
        author_email='jhansjakob@googlemail.com',
        description='',
        long_description='',
        ext_modules=[CMakeExtension(name, 'pyrlhook')],
        cmdclass=dict(build_ext=CMakeBuild),
        packages=['bin'],
        package_data={'bin': [
            'Win32/FunctionAddressGetter.exe',
            'Win32/RLHookDLL.dll',
            'x64/RLHookDLL.dll'
        ]},
        include_package_data=True,
        # packages=find_packages(),
        zip_safe=False,
        install_requires=['numpy']
    )
