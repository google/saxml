// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package ipaddr provides functions related to network addresses.
package ipaddr

import (
	"net"
	"os"
	"sync"

	log "github.com/golang/glog"
)

// localIP returns one of this machine's IP address reachable by other
// machines.
func localIP() net.IP {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		log.Errorf("No net ifac addresses. %v\n", err)
		return nil
	}
	ips := []net.IP{}
	for _, addr := range addrs {
		switch addr := addr.(type) {
		case *net.IPNet:
			ip := addr.IP
			if !ip.IsGlobalUnicast() {
				log.Infof("Skipping non-global IP address %s.\n", addr)
				continue
			}
			if len(ip) == net.IPv6len && ip[0] >= 0xfc {
				log.Infof("Skipping more RFC1918_GOOGLE IP address %s.\n", addr)
				continue
			}
			ips = append(ips, addr.IP)
		default:
			log.Infof("Skipping non-IP address %T %s.\n", addr, addr)
		}
	}
	if len(ips) == 0 {
		log.Errorf("No usable IP address.\n")
		return nil
	}
	for _, ip := range ips {
		log.Infof("IPNet address %s\n", ip.String())
	}
	return ips[0]
}

var myIPAddressMu sync.Mutex
var myIPAddress net.IP = nil

// MyIPAddr returns the ip address of this process reachable by others.
func MyIPAddr() net.IP {
	debugIpaddr := os.Getenv("SAX_DEBUG_IPADDR")
	if debugIpaddr == "1" {
		return net.IPv4(127, 0, 0, 1)
	}
	myIPAddressMu.Lock()
	defer myIPAddressMu.Unlock()
	if myIPAddress == nil {
		myIPAddress = localIP()
		if myIPAddress == nil {
			log.Fatalf("Unexpected. No usable IP address found.")
		}
	}
	return myIPAddress
}
