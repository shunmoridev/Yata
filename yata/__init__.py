# The one port Yata serves on (127.0.0.1 only). The page, its CSP and the allowed origins are tied to it; it lives
# here so the launcher can check the port without building the app (yata.api creates it on import).
PORT = 8765
