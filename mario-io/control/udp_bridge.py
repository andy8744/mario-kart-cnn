# control/udp_bridge.py

import socket
import json


class UDPBridge:
    def __init__(self, ip="192.168.64.2", port=5005):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, packet):
        self.sock.sendto(json.dumps(packet).encode("utf-8"), self.addr)
