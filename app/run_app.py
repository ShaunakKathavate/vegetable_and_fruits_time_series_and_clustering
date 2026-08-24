"""Launcher for the Streamlit app.

Needed on this machine because a corrupted certificate in the Windows certificate store makes
`ssl.SSLContext.load_default_certs()` raise `SSLError: [ASN1: NOT_ENOUGH_DATA]` -- unconditionally,
regardless of SSL_CERT_FILE/SSL_CERT_DIR -- the moment `tornado` (a streamlit dependency) is
imported. This has nothing to do with this project; it's a broken cert sitting in the OS store.
Patching `_load_windows_store_certs` to skip whichever cert fails to parse, applied before
streamlit/tornado import, avoids it without touching the OS certificate store itself.

Usage:
    python app/run_app.py [any streamlit run args, e.g. --server.port 8501]
"""

import ssl
import sys

_original_load_windows_store_certs = ssl.SSLContext._load_windows_store_certs


def _safe_load_windows_store_certs(self, storename, purpose):
    try:
        _original_load_windows_store_certs(self, storename, purpose)
    except ssl.SSLError:
        pass


ssl.SSLContext._load_windows_store_certs = _safe_load_windows_store_certs

from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", __file__.replace("run_app.py", "Home.py")] + sys.argv[1:]
    sys.exit(main())
