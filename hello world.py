import socket 
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('127.0.0.1', 9999))
print('Binding UDP on 9999...')
while True:
    data, address = s.recvfrom(1024)
    print('Recieve from %s:%s' % address)
    print('just a test')
    s.sendto(b'hello, %s' % data, address)

