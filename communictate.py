#!/home/orangepi/miniconda3/bin/python

import concurrent.futures
import serial
import threading
import time
import socket
import struct

import concurrent.futures
import threading
import socket
import time

# node = Node('process_node','serial',process_fun  )
class Node:
    def __init__(self, name, mode, f, ip=None, port=None, com='/dev/ttyS0', baudrate=460800, timeout=2):
        self.name = name
        self.neighbors = {}
        self.lock = threading.Lock()
        self.process_function = f
        self.mode = mode
        self.has_data=False 

        # Create a ThreadPoolExecutor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        # Initialize according to mode
        if mode == 'network':
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.sock.bind((ip, port))
            self.broadcast_ip = '<broadcast>'
            self.port = port
            self.ip = ip
            # Start threads
            self.executor.submit(self.broadcast)
            self.executor.submit(self.receive)
            
        elif mode == 'serial':
            self.ser = serial.Serial()
            self.ser.baudrate = baudrate
            self.ser.timeout = timeout
            if com is not None:
                self.ser.port = com
                self.ser.open()

            # Start threads
            self.executor.submit(self.read_serial)
            # self.executor.submit(self.broadcast_serial)

        self.remote_ip = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def broadcast(self):
        while True:
            msg = "Node:" + self.name + "," + \
                str(self.ip) + "," + str(self.port)
            self.lock.acquire()
            try:
                self.sock.sendto(msg.encode(), (self.broadcast_ip, self.port))
            finally:
                self.lock.release()
            time.sleep(2)

    def receive(self):
        while True:
          try:
            data, addr = self.sock.recvfrom(1024)
            if data[0]==0x55 and len(self.neighbors )==0:
                self.neighbors['esp_diff'] = addr
                self.neighbors['debug_node'] = ('192.168.2.46',28288)
                # print(addr) 
                self.has_data=True
            if data.startswith(b'Node'):
                node_name, node_ip, node_port = data.decode().split(':')[1].split(',')
                self.lock.acquire()
                try:
                    self.neighbors[node_name] = (node_ip, int(node_port))
                    self.has_data=True
                finally:
                    self.lock.release()
            else:
                self.process_function(data)
          except Exception as e:
              print(e) 

    def read_serial(self):
        if not self.ser.is_open:
            print("Serial port is not open. Can't read.")
            return
        while True:
          try:
            
            data = self.ser.read_until(b'\r\t')

            # if int(data[0])==0x55 and int(data[1])==0xaa:
            #     data_len=struct.unpack('h',data[2:])[0] 
            #     data=self.ser.read(data_len) 
                
            # print("data",(data)) 
            if data.endswith(b'\r\t') :
                data=data[:-2] 
            
            self.process_function(data ) 
            self.has_data=True 
                # We pass the serial object instead of an address 
          except Exception as e:
              self.ser.flushInput()
              print(e) 

    def broadcast_serial(self):
        while True:
            msg = "Node:" + self.name+","
            self.lock.acquire()
            try:
                self.ser.write(msg.encode())
            except Exception as e:
                print(e) 
            finally:
                self.lock.release()
            time.sleep(2)
    def send_to_serial(self, message):
        # print(self.neighbors) 
        if self.mode == 'serial':
            ser = self.ser
            try:
                ser.write(message)
            except serial.SerialException as e:
                print(f"Error occured while sending data: {e}")
    def send_to_node(self, node_name, message):
        # print(self.neighbors) 
        if node_name not in self.neighbors:
            print(f"No such node: {node_name}")
            return

        self.lock.acquire()
        try:
            identifier = self.neighbors[node_name]
        finally:
            self.lock.release()

        if self.mode == 'network':
            ip, port = identifier
            try:
                self.sock.sendto(message, (ip, port))
            except socket.error as e:
                print(f"Error occured while sending data: {e}")
        elif self.mode == 'serial':
            ser = identifier
            try:
                ser.write(message)
            except serial.SerialException as e:
                print(f"Error occured while sending data: {e}")

    def close(self):
        self.executor.shutdown(wait=True)


def get_ip():

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
    str = s.getsockname()[0]
    s.close()
    return str


lock = threading.RLock()
ser = None
has_data = False
mutex = threading.Lock()  # 创建锁


def check_sum(data):
    sum = 0
    for i in range(len(data[:-4])):
        sum += data[i]
    # print(sum)
    return sum


 


# def process_fun(data, addr, node):
#     global t1, t2, data_win
#     # print(data,addr
#     # t1=time.time()
#     # f=1/(t1-t2)
#     # t2=t1
#     global data_line, data_tosave, once, new_data, has_data
#     global recive_data, data_line

#     # print(data)
#     if not data.startswith(b'Node'):
#         # try:
#         if(check_sum(data) != struct.unpack('I', data[-4:])[0]):
#             print('check sum error')
#             return
#         data_line = struct.unpack('BBIffffffffffI', data[:-4])[2:]

#         data_line = list(data_line)
#         node.remote_ip = addr[0]
#         mutex.acquire()
#         data_win.append(data_line)
#         if(len(data_win) > 10):
#             data_win.pop(0)
#         mutex.release()
#         has_data = True


t1=time.time()
def process_fun(data):
    global t1  
    dt=time.time()-t1
    t1=time.time()
    # print(dt*1000) 


if __name__=="__main__":
    # node = Node('process_node','network',process_fun,ip='
    node =Node ('process_node','serial',process_fun,com='/dev/ttyS0',baudrate=460800,timeout=0.2)

# while 1:
#     time.sleep(1)
