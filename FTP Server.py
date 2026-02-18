from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import os
import time

class ReolinkFTPHandler(FTPHandler):
    def on_file_received(self, file):
        
        print(f"\n[NEW FILE UPLOADED] {file}")

        try:
            # Get file information immediately (no delay needed)
            if os.path.exists(file):
                stat = os.stat(file)

                size_mb = stat.st_size / (1024 * 1024)          # size in MB for readability
                mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime))

                print(f"  └─ Size:     {stat.st_size:,} bytes ({size_mb:.2f} MB)")
                print(f"  └─ Modified: {mod_time}")
                print(f"  └─ Created:  {create_time} (Windows creation time)")

                # Optional extras you can add:
                # - File extension check
                if file.lower().endswith(('.jpg', '.jpeg')):
                    print("  → This is a snapshot!")
                elif file.lower().endswith('.mp4'):
                    print("  → This is a video clip!")
            else:
                print("  Warning: File disappeared right after upload?!")

        except Exception as e:
            print(f"  Error getting info for {file}: {e}")


def main():
    # Create a dummy authorizer for "virtual" users
    authorizer = DummyAuthorizer()
    authorizer.add_user("reolink", "H@rryP0tter", r"C:\reolink", perm="elradfmw")

    # Instantiate FTP handler
    handler = ReolinkFTPHandler
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