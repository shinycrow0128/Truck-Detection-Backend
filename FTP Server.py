from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

def main():
    # Create a dummy authorizer for "virtual" users
    authorizer = DummyAuthorizer()
    authorizer.add_user("reolink", "H@rryP0tter", r"C:\reolink", perm="elradfmw")

    # Instantiate FTP handler
    handler = FTPHandler
    handler.authorizer = authorizer

    # Optional: customize banner or other settings
    handler.banner = "Reolink FTP server ready."

    # Listen on all interfaces, port 21
    address = ('0.0.0.0', 21)

    server = FTPServer(address, handler)
    server.max_cons = 50
    server.max_cons_per_ip = 5

    print("FTP server started on port 21...")
    server.serve_forever()

if __name__ == '__main__':
    main()