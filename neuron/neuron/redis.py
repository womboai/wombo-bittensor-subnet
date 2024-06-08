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

from urllib.parse import urlparse


def parse_redis_value(value: str | None, t: type):
    if value is None:
        return t()

    return t(value)


def parse_redis_uri(uri: str):
    url = urlparse(uri)

    if url.scheme == "redis":
        ssl = False
    elif url.scheme == "rediss":
        ssl = True
    else:
        raise RuntimeError(f"Invalid Redis scheme {url.scheme}")

    if url.path:
        path_db = url.path[1:]

        if not path_db:
            db = 0
        else:
            db = int(path_db)
    else:
        db = 0

    if not url.username or url.password:
        username = url.username
        password = url.password
    else:
        username = None
        password = url.username

    return {
        "host": url.hostname,
        "port": url.port,
        "db": db,
        "password": password,
        "ssl": ssl,
        "username": username,
    }
