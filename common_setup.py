# Used as symlink to individual projects, */common_setup.py should not be edited,
# instead the root version of this file should be edited and changes should be reflected in all dependant files

import os.path
import re
from pathlib import Path


def read_requirements(path: str):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        processed_requirements = []

        for req in requirements:
            # For git or other VCS links
            if req.startswith("file:"):
                path = os.path.join(os.getcwd(), req[len("file:"):])
                package_name = os.path.basename(path)
                uri = Path(path).as_uri()

                processed_requirements.append(f"wombo-bittensor-subnet-{package_name}@{uri}")
            elif req.startswith("git+") or "@" in req:
                pkg_name = re.search(r"(#egg=)([\w\-_]+)", req)
                if pkg_name:
                    processed_requirements.append(pkg_name.group(2))
                else:
                    # You may decide to raise an exception here,
                    # if you want to ensure every VCS link has an #egg=<package_name> at the end
                    continue
            else:
                processed_requirements.append(req)
        return processed_requirements
