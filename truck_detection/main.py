from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.servers import FTPServer

from truck_detection.ftp_handler import ReolinkFTPHandler


def main():
    authorizer = DummyAuthorizer()
    authorizer.add_user("reolink", "H@rryP0tter", r"C:\reolink", perm="elradfmw")

    handler = ReolinkFTPHandler
    handler.authorizer = authorizer
    handler.banner = "Reolink FTP server ready."

    address = ("0.0.0.0", 21)
    server = FTPServer(address, handler)
    server.max_cons = 50
    server.max_cons_per_ip = 5

    print("FTP server started on port 21...")
    print("Waiting for Reolink uploads → truck analysis → Supabase upload (when truck confirmed)")
    server.serve_forever()


if __name__ == "__main__":
    main()
