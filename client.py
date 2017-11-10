import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for data in ['Mick', 'Trac' ,'Sara']:
    s.sendto(data.encode('ascii'), ('127.0.0.1', 9999))
    print(s.recv(1024).decode('utf-8'))