# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 16:27:35 2025

@author: m.ghorbani
"""

import os
import sys
import socket
import ipaddress
import ctypes

# =============================
# COMPANY NETWORKS (ONLY EDIT HERE)
# =============================
ALLOWED_NETWORKS = [
    ipaddress.ip_network("172.18.80.0/20"),   # Corporate / VPN
    ipaddress.ip_network("192.168.1.0/24"),   # Office LAN
    ipaddress.ip_network("192.168.80.0/24")   # servers
]

# =============================
# LOW LEVEL CHECKS
# =============================
def _get_all_ipv4():
    ips = set()
    hostname = socket.gethostname()

    try:
        for info in socket.getaddrinfo(hostname, None):
            if info[0] == socket.AF_INET:
                ips.add(info[4][0])
    except Exception:
        pass

    return ips

def _check_ip_range():
    ips = _get_all_ipv4()
    for ip in ips:
        ip_obj = ipaddress.ip_address(ip)
        for net in ALLOWED_NETWORKS:
            if ip_obj in net:
                return
    raise RuntimeError("Unauthorized network")

def _anti_debug():
    if sys.gettrace():
        raise RuntimeError("Debugger detected")
    try:
        if ctypes.windll.kernel32.IsDebuggerPresent():
            raise RuntimeError("Debugger detected")
    except Exception:
        pass
# =============================
# PUBLIC ENTRY POINT
# =============================
def enforce():
    try:
        _anti_debug()
        _check_ip_range()
    except Exception:
        # silent hard exit
        os._exit(1)

# =============================
# AUTO-RUN ON IMPORT (STRONGER)
# =============================
enforce()