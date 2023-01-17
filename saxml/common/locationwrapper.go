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

// locationwrapper exports functions from the location package for Cgo.
package main

import (
	"C"
	"unsafe"
)

import (
	"context"

	"google.golang.org/protobuf/proto"
	"saxml/common/location"
	_ "saxml/common/platform/register" // registers a platform

	pb "saxml/protobuf/admin_go_proto_grpc"
)

// join calls Join with a background context.
func join(saxCell, ipPort, debugAddr string, specs *pb.ModelServer, adminPort int) error {
	// The caller on the C++ side is expected to call Join in a local, non-RPC thread on model server
	// start (as opposed to in an RPC request handler), so there isn't a C++ context to pass in and
	// transport here through RegisterContextTransport. For simplicity, we can create a Go background
	// context here instead of creating an empty C++ context and passing it in.
	return location.Join(context.Background(), saxCell, ipPort, debugAddr, specs, adminPort)
}

//export sax_join
func sax_join(saxCellPtr *C.char, saxCellSize C.int, ipPortPtr *C.char, ipPortSize C.int, debugAddrPtr *C.char, debugAddrSize C.int, specsPtr unsafe.Pointer, specsSize C.int, adminPort int) *C.char {
	saxCell := C.GoStringN(saxCellPtr, saxCellSize)
	ipPort := C.GoStringN(ipPortPtr, ipPortSize)
	debugAddr := C.GoStringN(debugAddrPtr, debugAddrSize)
	serializedSpecs := C.GoBytes(specsPtr, specsSize)
	specs := &pb.ModelServer{}
	if err := proto.Unmarshal(serializedSpecs, specs); err != nil {
		return C.CString("invalid input serialized specs: " + err.Error())
	}
	if err := join(saxCell, ipPort, debugAddr, specs, adminPort); err != nil {
		return C.CString(err.Error())
	}
	return C.CString("")
}

func main() {}
