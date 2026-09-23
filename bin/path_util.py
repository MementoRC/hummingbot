#!/usr/bin/python

if "hummingbot-dist" in __file__:
    # Dist environment.
    import os
    import sys
    sys.path.append(sys.path.pop(0))
    sys.path.insert(0, os.getcwd())

    import hummingbot
    hummingbot.set_prefix_path(os.getcwd())
else:
    # Dev environment.
    import os
    import sys
    sys.path.insert(0, os.path.realpath(os.path.join(__file__, "../../")))
