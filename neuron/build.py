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

from os import listdir, PathLike
from os.path import isfile, join
from pathlib import Path

import grpc_tools.protoc
from more_itertools import flatten


def list_all_files(directory: PathLike | str):
    return flatten(
        [
            [Path(join(directory, file)).absolute()]
            if isfile(file)
            else list_all_files(join(directory, file))
            for file in listdir(directory)
        ]
    )


def build(_setup_kwargs):
    project_folder = Path(__file__).parent.absolute()
    root_folder = project_folder.parent.absolute()
    protos_directory = project_folder / "protos"

    proto_files = [protos_directory / file for file in listdir(protos_directory)]

    args = [
        "--proto_path",
        root_folder,
        *proto_files,
        "--python_out",
        project_folder,
        "--grpc_python_out",
        project_folder,
    ]

    grpc_tools.protoc.main(args)
