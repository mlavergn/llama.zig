"""The upstream pin the generators stamp their output with.

**Not a port.** Read from `build/upstream.zig`, the one place the pin is
written down, so a regenerated file cites the commit it was generated
from. The generators once each carried `COMMIT = "c1d0e7a00"`: regenerating
one after the pin moved would have cited new line numbers at the old
commit, every one of them stale.
"""

import os
import re

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "upstream.zig")


def _field(name):
    m = re.search(r'pub const %s = "([^"]+)";' % name, open(_PATH).read())
    if not m:
        raise SystemExit("upstream_pin: no `%s` in build/upstream.zig" % name)
    return m.group(1)


TAG = _field("tag")
COMMIT = _field("commit")
