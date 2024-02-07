import os.path
import re
from pathlib import Path


def read_requirements(path):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        processed_requirements = []

        for req in requirements:
            # For git or other VCS links
            if req.startswith("file:"):
                path = os.path.join(os.getcwd(), req[len("file:"):])
                package_name = os.path.basename(path)
                uri = Path(path).as_uri()
                requirement = f"wombo-bittensor-subnet-{package_name}@{uri}"

                processed_requirements.append(requirement)
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
