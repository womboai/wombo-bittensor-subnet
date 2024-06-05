#  The MIT License (MIT)
#  Copyright © 2023 Yuma Rao
#  Copyright © 2024 WOMBO
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#
#

from itertools import chain
from os import listdir, PathLike
from os.path import isfile, join
from pathlib import Path
from typing import Any

import grpc_tools.protoc
from setuptools.command.build_py import build_py


def list_all_files(directory: PathLike | str):
    return chain.from_iterable(
        [
            [str(Path(join(directory, file)).absolute())]
            if isfile(join(directory, file))
            else list_all_files(join(directory, file))
            for file in listdir(directory)
        ]
    )


class Build(build_py):
    def run(self):
        project_folder = Path(__file__).parent.absolute()
        root_folder = project_folder.parent.absolute()
        protos_directory = project_folder / "protos"

        google_std = Path(grpc_tools.__file__).parent / "_proto"

        args = [
            grpc_tools.protoc.__file__,

            "--proto_path", str(root_folder),
            f"-I{google_std}",

            "--python_out", str(project_folder),
            "--pyi_out", str(project_folder),
            "--grpc_python_out", str(project_folder),

            *list_all_files(protos_directory),
        ]

        exit_code = grpc_tools.protoc.main(args)

        if exit_code:
            raise RuntimeError(f"grpc_tools.protoc returned exit code {exit_code}")

        super().run()


def build(setup_kwargs: dict[str, Any]):
    setup_kwargs.update(
        {
            "cmdclass": {
                "build_py": Build
            }
        }
    )
