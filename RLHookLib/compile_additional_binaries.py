import os
import sys
import subprocess
from enum import Enum
from typing import Optional


class Architecture(Enum):
    Win32 = 'Win32'
    x64 = 'x64'


def build_cmake_project(
        source_dir: str,
        build_dir: str,
        out_dir: str,
        architecture: Optional[Architecture],
        target: Optional[str] = None,
        additional_binaries: bool = False,
        debug: bool = False
):

    out_dir = os.path.abspath(out_dir)

    if architecture is not None:
        build_dir = os.path.join(build_dir, architecture.value)
        out_dir = os.path.join(out_dir, architecture.value)

    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    cfg = 'Debug' if debug else 'Release'
    cmake_args = [
        f'-B{build_dir}',
        f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={out_dir}',
        f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={out_dir}',
        f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={out_dir}',
        f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={out_dir}',
        f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{cfg.upper()}={out_dir}',
        f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{cfg.upper()}={out_dir}',
        #f'-DPYTHON_EXECUTABLE={sys.executable}',
        #f'-DCMAKE_BUILD_TYPE={cfg.upper()}'
    ]
    if additional_binaries:
        cmake_args += ['-DONLY_ADDITIONAL_BINARIES=ON']
    if architecture is not None:
        cmake_args += ['-A', architecture.value]

    build_args = ['--config', cfg]
    if target is not None:
        build_args += ['--target', target]
    build_args += ['--', '/m']

    subprocess.check_call(['cmake', source_dir] + cmake_args)  # env=env
    subprocess.check_call(['cmake', '--build', build_dir] + build_args)


if __name__ == '__main__':

    projects_to_build = [
        ('RLHookDLL', Architecture.Win32),
        ('RLHookDLL', Architecture.x64),
        ('FunctionAddressGetter', Architecture.Win32),
    ]

    for target, architecture in projects_to_build:
        build_cmake_project('.', f'build', f'bin', architecture, target, additional_binaries=True)
