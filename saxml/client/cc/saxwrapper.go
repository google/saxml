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

// saxwrapper wraps sax client library using cgo to provide c/c++ API.
package main

/*
#include <stdbool.h>

typedef void (*generate_callback)(void* cbCtx, void* outData, int outSize);

static inline void generate_callback_bridge(generate_callback cb, void* cbCtx, void* outData, int outSize) {
	cb(cbCtx, outData, outSize);
}
*/
import "C"

import (
	"context"
	"io"
	rcgo "runtime/cgo"
	"time"
	"unsafe"

	log "github.com/golang/glog"
	"google.golang.org/protobuf/proto"
	"saxml/client/go/sax"
	"saxml/client/go/saxadmin"
	"saxml/common/errors"
	"saxml/common/naming"
	_ "saxml/common/platform/register" // registers a platform

	apb "saxml/protobuf/admin_go_proto_grpc"
	ampb "saxml/protobuf/audio_go_proto_grpc"
	cpb "saxml/protobuf/common_go_proto"
	cmpb "saxml/protobuf/custom_go_proto_grpc"
	lmpb "saxml/protobuf/lm_go_proto_grpc"
	mmpb "saxml/protobuf/multimodal_go_proto_grpc"
	vmpb "saxml/protobuf/vision_go_proto_grpc"
)

// Dummy return for when model creation fails; value doesn't matter.
const nilModel = C.long(42)

func protoOptionToSetter(options *cpb.ExtraInputs) []sax.ModelOptionSetter {
	var extra = []sax.ModelOptionSetter{}
	if options != nil {
		for name, val := range options.GetItems() {
			extra = append(extra, sax.WithExtraInput(name, val))
		}
		for name, val := range options.GetTensors() {
			extra = append(extra, sax.WithExtraInputTensor(name, val.GetValues()))
		}
		for name, val := range options.GetStrings() {
			extra = append(extra, sax.WithExtraInputString(name, val))
		}
	}
	return extra
}

// buildReturnError is a helper function for model methods that return an error.
// "outErrMsgData" is for error message. They are ascii printable characters so size is not needed. It's empty when there is no error.
// "outErrCode" is the canonical error code for the error. It's 0 when there is no error.
func buildReturnError(outErrMsgData **C.char, outErrCode *C.int, err error) {
	errCode := errors.Code(err)
	if errCode != 0 {
		*outErrMsgData = C.CString(err.Error())
	}
	*outErrCode = C.int(errCode)
}

// buildReturnValues is a helper function for model methods that return both returned values (such as score or text) and error.
// "outValueData" and "outValueSize" are for returned value. Size is needed since outData might contain "\0".
// "outErrMsgData" is for error message. They are ascii printable characters so size is not needed. It's empty when there is no error.
// "outErrCode" is the canonical error code for the error. It's 0 when there is no error.
func buildReturnValues(outValueData **C.char, outValueSize *C.int, outErrMsgData **C.char, outErrCode *C.int, value *[]byte, err error) {
	if int(errors.Code(err)) == 0 {
		*outValueData = C.CString(string(*value))
		*outValueSize = C.int(len(*value))
	}
	buildReturnError(outErrMsgData, outErrCode, err)
}

// createContextWithTimeout creates a timeout context if "timeout" is positive.
func createContextWithTimeout(timeout C.float) (context.Context, context.CancelFunc) {
	numSeconds := float64(timeout)
	if numSeconds > 0.0 {
		return context.WithTimeout(context.Background(), time.Duration(numSeconds*float64(time.Second)))
	}
	return context.Background(), nil
}

//////////////////////////////////////////////////////////////////////////
// Model creation methods
//////////////////////////////////////////////////////////////////////////

//export go_release_model
func go_release_model(in C.long) {
	h := rcgo.Handle(in)
	h.Delete()
}

//export go_create_model_with_config
func go_create_model_with_config(idData *C.char, idSize C.int, numConn C.int, proxyAddrData *C.char, proxyAddrSize C.int, failFast C.bool, out *C.long, errMsg **C.char, errCode *C.int) {
	id := C.GoStringN(idData, idSize)
	proxyAddr := C.GoStringN(proxyAddrData, proxyAddrSize)
	m, err := sax.Open(id, sax.WithNumConn(int(numConn)), sax.WithProxy(proxyAddr), sax.WithFailFast(bool(failFast)))
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}
	*out = C.long(rcgo.NewHandle(m))
	*errCode = 0
}

//export go_create_model
func go_create_model(idData *C.char, idSize C.int, out *C.long, errMsg **C.char, errCode *C.int) {
	id := C.GoStringN(idData, idSize)
	m, err := sax.Open(id)
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}
	*out = C.long(rcgo.NewHandle(m))
	*errCode = 0
}

//export go_create_am
func go_create_am(in C.long, out *C.long) {
	m := rcgo.Handle(C.long(in)).Value().(*sax.Model)
	if m == nil {
		*out = nilModel
		return
	}
	*out = C.long(rcgo.NewHandle(m.AM()))
}

//export go_create_cm
func go_create_cm(in C.long, out *C.long) {
	m := rcgo.Handle(C.long(in)).Value().(*sax.Model)
	if m == nil {
		*out = nilModel
		return
	}
	*out = C.long(rcgo.NewHandle(m.CM()))
}

//export go_create_mm
func go_create_mm(in C.long, out *C.long) {
	m := rcgo.Handle(C.long(in)).Value().(*sax.Model)
	if m == nil {
		*out = nilModel
		return
	}
	*out = C.long(rcgo.NewHandle(m.MM()))
}

//export go_create_lm
func go_create_lm(in C.long, out *C.long) {
	m := rcgo.Handle(C.long(in)).Value().(*sax.Model)
	if m == nil {
		*out = nilModel
		return
	}
	*out = C.long(rcgo.NewHandle(m.LM()))
}

//export go_create_vm
func go_create_vm(in C.long, out *C.long) {
	m := rcgo.Handle(C.long(in)).Value().(*sax.Model)
	if m == nil {
		*out = nilModel
		return
	}
	*out = C.long(rcgo.NewHandle(m.VM()))
}

//////////////////////////////////////////////////////////////////////////
// Admin methods
//////////////////////////////////////////////////////////////////////////

func openAdmin(id string) (string, *saxadmin.Admin, error) {
	modelID, err := naming.NewModelFullName(id)
	if err != nil {
		return "", nil, err
	}
	return modelID.ModelFullName(), saxadmin.Open(modelID.CellFullName()), nil
}

//export go_publish
func go_publish(idData *C.char, idSize C.int, modelPathData *C.char, modelPathSize C.int,
	checkpointPathData *C.char, checkpointPathSize C.int, numReplicas C.int, timeout C.float,
	numOverrides C.int, overrideKeys **C.char, overrideKeySizes *C.int, overrideValues **C.char,
	overrideValueSizes *C.int, errMsg **C.char, errCode *C.int) {
	id := C.GoStringN(idData, idSize)
	modelID, admin, err := openAdmin(id)
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}

	modelPath := C.GoStringN(modelPathData, modelPathSize)
	checkpointPath := C.GoStringN(checkpointPathData, checkpointPathSize)
	overrides := make(map[string]string)
	goOverrideKeys := unsafe.Slice(overrideKeys, numOverrides)
	goOverrideKeySizes := unsafe.Slice(overrideKeySizes, numOverrides)
	goOverrideValues := unsafe.Slice(overrideValues, numOverrides)
	goOverrideValueSizes := unsafe.Slice(overrideValueSizes, numOverrides)
	for i := 0; i < int(numOverrides); i++ {
		key := C.GoStringN(goOverrideKeys[i], goOverrideKeySizes[i])
		value := C.GoStringN(goOverrideValues[i], goOverrideValueSizes[i])
		overrides[key] = value
	}
	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}
	err = admin.Publish(ctx, modelID, modelPath, checkpointPath, int(numReplicas), overrides)
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}
}

//export go_unpublish
func go_unpublish(idData *C.char, idSize C.int, timeout C.float, errMsg **C.char, errCode *C.int) {
	id := C.GoStringN(idData, idSize)
	modelID, admin, err := openAdmin(id)
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}
	err = admin.Unpublish(ctx, modelID)
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}
}

//export go_update
func go_update(idData *C.char, idSize C.int, modelPathData *C.char, modelPathSize C.int, checkpointPathData *C.char, checkpointPathSize C.int, numReplicas C.int, timeout C.float, errMsg **C.char, errCode *C.int) {
	id := C.GoStringN(idData, idSize)
	modelID, admin, err := openAdmin(id)
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}

	modelPath := C.GoStringN(modelPathData, modelPathSize)
	checkpointPath := C.GoStringN(checkpointPathData, checkpointPathSize)
	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	publishedModel, err := admin.List(ctx, modelID)
	if err != nil || publishedModel == nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}

	err = admin.Update(ctx, &apb.Model{
		ModelId:              modelID,
		ModelPath:            modelPath,
		CheckpointPath:       checkpointPath,
		RequestedNumReplicas: int32(numReplicas),
		Overrides:            publishedModel.Model.Overrides,
	})
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}
}

//export go_list
func go_list(idData *C.char, idSize C.int, timeout C.float, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	id := C.GoStringN(idData, idSize)
	modelID, admin, err := openAdmin(id)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}
	publishedModel, err := admin.List(ctx, modelID)
	if err != nil || publishedModel == nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	content, err := proto.Marshal(publishedModel)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_list_all
func go_list_all(idData *C.char, idSize C.int, timeout C.float, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	id := C.GoStringN(idData, idSize)
	admin := saxadmin.Open(id)
	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}
	listResp, err := admin.ListAll(ctx)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	content, err := proto.Marshal(listResp)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_wait_for_ready
func go_wait_for_ready(idData *C.char, idSize C.int, numReplicas C.int, timeout C.float, errMsg **C.char, errCode *C.int) {
	id := C.GoStringN(idData, idSize)
	modelID, admin, err := openAdmin(id)
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}
	err = admin.WaitForReady(ctx, modelID, int(numReplicas))
	if err != nil {
		*errMsg = C.CString(err.Error())
		*errCode = C.int(int32(errors.Code(err)))
		return
	}
}

//export go_stats
func go_stats(idData *C.char, idSize C.int, timeout C.float, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	id := C.GoStringN(idData, idSize)
	isModelFullName := naming.ValidateModelFullName(id) == nil
	var modelID string
	var admin *saxadmin.Admin
	if isModelFullName {
		var err error
		modelID, admin, err = openAdmin(id)
		if err != nil {
			buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
			return
		}
	} else {
		admin = saxadmin.Open(id)
		modelID = ""
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}
	listResp, err := admin.Stats(ctx, modelID)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	content, err := proto.Marshal(listResp)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_get_acl
func go_get_acl(nameData *C.char, nameSize C.int, timeout C.float, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	name := C.GoStringN(nameData, nameSize)
	saxCell, _, err := naming.SaxCell(name)
	if err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}

	admin := saxadmin.Open(saxCell)
	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	acl, err := admin.GetACL(ctx, name)
	if err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}

	*outData = C.CString(acl)
	*outSize = C.int(len(acl))
}

//export go_get_sax_model_data_method_acls
func go_get_sax_model_data_method_acls(nameData *C.char, nameSize C.int, timeout C.float, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	name := C.GoStringN(nameData, nameSize)
	saxCell, _, err := naming.SaxCell(name)
	if err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}

	admin := saxadmin.Open(saxCell)
	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	acls, err := admin.GetSaxModelDataMethodACLs(ctx, name)
	if err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}

	content, err := proto.Marshal(acls)
	if err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_set_acl
func go_set_acl(nameData *C.char, nameSize C.int, aclData *C.char, aclSize C.int, timeout C.float, errMsg **C.char, errCode *C.int) {
	name := C.GoStringN(nameData, nameSize)
	saxCell, _, err := naming.SaxCell(name)
	if err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}

	admin := saxadmin.Open(saxCell)
	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	acl := C.GoStringN(aclData, aclSize)
	err = admin.SetACL(ctx, name, acl)
	if err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}
}

//export go_set_sax_model_data_method_acl
func go_set_sax_model_data_method_acl(nameData *C.char, nameSize C.int, methodData *C.char, methodSize C.int, aclData *C.char, aclSize C.int, timeout C.float, errMsg **C.char, errCode *C.int) {
	name := C.GoStringN(nameData, nameSize)
	saxCell, _, err := naming.SaxCell(name)
	if err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}

	admin := saxadmin.Open(saxCell)
	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	method := C.GoStringN(methodData, methodSize)
	acl := C.GoStringN(aclData, aclSize)
	err = admin.SetSaxModelDataMethodACL(ctx, name, method, acl)
	if err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}
}

//////////////////////////////////////////////////////////////////////////
// Audio model methods
//////////////////////////////////////////////////////////////////////////

//export go_recognize
func go_recognize(ci C.long, timeout C.float, audioBytesData *C.char, audioBytesSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsgData **C.char, errCode *C.int) {
	am := rcgo.Handle(ci).Value().(*sax.AudioModel)
	if am == nil {
		// This is not expected.
		log.Fatalf("recognize() called on nil audio model.")
	}

	options := &cpb.ExtraInputs{}
	optionsStr := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	if err := proto.Unmarshal(optionsStr, options); err != nil {
		buildReturnValues(outData, outSize, errMsgData, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	audioBytes := C.GoBytes(unsafe.Pointer(audioBytesData), audioBytesSize)
	res, err := am.Recognize(ctx, audioBytes, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsgData, errCode, nil, err)
		return
	}
	ret := &ampb.AsrResponse{}
	for _, v := range res {
		item := &ampb.AsrHypothesis{
			Text:  v.Text,
			Score: v.Score,
		}
		ret.Hyps = append(ret.GetHyps(), item)
	}

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsgData, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsgData, errCode, &content, nil)
}

//////////////////////////////////////////////////////////////////////////
// Language model methods
//////////////////////////////////////////////////////////////////////////

//export go_score
func go_score(ptr C.long, timeout C.float, requestData *C.char, requestSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, queryCostTpuMs *C.int, errMsg **C.char, errCode *C.int) {
	lm := rcgo.Handle(ptr).Value().(*sax.LanguageModel)
	if lm == nil {
		// This is not expected.
		log.Fatalf("score() called on nil language model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	requestByte := C.GoBytes(unsafe.Pointer(requestData), requestSize)
	request := &lmpb.ScoreRequest{}
	if err := proto.Unmarshal(requestByte, request); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	setters := protoOptionToSetter(options)
	queryCost := sax.QueryCost{}
	setters = append(setters, sax.WithQueryCost(&queryCost))
	score, err := lm.Score(ctx, request.GetPrefix(), request.GetSuffix(), setters...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	resp := &lmpb.ScoreResponse{}
	for _, one := range score {
		resp.Logp = append(resp.GetLogp(), one)
	}
	content, err := proto.Marshal(resp)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_generate
func go_generate(ptr C.long, timeout C.float, textData *C.char, textSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, queryCostTpuMs *C.int, errMsg **C.char, errCode *C.int) {
	lm := rcgo.Handle(ptr).Value().(*sax.LanguageModel)
	if lm == nil {
		// This is not expected.
		log.Fatalf("generate() called on nil language model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	text := C.GoStringN(textData, textSize)
	setters := protoOptionToSetter(options)
	queryCost := sax.QueryCost{}
	setters = append(setters, sax.WithQueryCost(&queryCost))
	res, err := lm.Generate(ctx, text, setters...)
	if queryCostTpuMs != nil {
		*queryCostTpuMs = C.int(queryCost.TpuMs)
	}
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	ret := &lmpb.GenerateResponse{}
	for _, v := range res {
		item := &lmpb.DecodedText{
			Text:  v.Text,
			Score: v.Score,
		}
		ret.Texts = append(ret.GetTexts(), item)
	}
	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_generate_stream
func go_generate_stream(ptr C.long, timeout C.float, textData *C.char, textSize C.int, optionsData *C.char, optionsSize C.int, cb C.generate_callback, cbCtx unsafe.Pointer, queryCostTpuMs *C.int, errMsg **C.char, errCode *C.int) {
	lm := rcgo.Handle(ptr).Value().(*sax.LanguageModel)
	if lm == nil {
		// This is not expected.
		log.Fatalf("streaming generate() called on nil language model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnError(errMsg, errCode, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	text := C.GoStringN(textData, textSize)
	setters := protoOptionToSetter(options)
	queryCost := sax.QueryCost{}
	setters = append(setters, sax.WithQueryCost(&queryCost))
	ch := lm.GenerateStream(ctx, text, setters...)
	if queryCostTpuMs != nil {
		*queryCostTpuMs = C.int(queryCost.TpuMs)
	}

	for res := range ch {
		err := res.Err
		switch err {
		case nil:
			ret := &lmpb.GenerateStreamResponse{}
			for _, v := range res.Items {
				item := &lmpb.GenerateStreamItem{
					Text:      v.Text,
					PrefixLen: int32(v.PrefixLen),
					Scores:    v.Scores,
				}
				ret.Items = append(ret.GetItems(), item)
			}
			content, err := proto.Marshal(ret)
			if err != nil {
				log.Fatal("streaming generate() fails to serialize return value")
			}
			outData := C.CBytes(content) // freed by C caller
			// For normal results, send a non-nil output.
			C.generate_callback_bridge(cb, cbCtx, outData, C.int(len(content)))
		case io.EOF:
			// On EOF, send a nil output to let the C++ wrapper translate it to a "last" bool.
			C.generate_callback_bridge(cb, cbCtx, nil, 0)
		default:
			// For any other error, end streaming with an error.
			buildReturnError(errMsg, errCode, err)
			return
		}
	}

	// End streaming with no error.
	buildReturnError(errMsg, errCode, nil)
}

//export go_lm_embed
func go_lm_embed(ptr C.long, timeout C.float, textData *C.char, textSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, queryCostTpuMs *C.int, errMsg **C.char, errCode *C.int) {
	lm := rcgo.Handle(ptr).Value().(*sax.LanguageModel)
	if lm == nil {
		// This is not expected.
		log.Fatalf("embed() called on nil language model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	text := C.GoStringN(textData, textSize)
	setters := protoOptionToSetter(options)
	queryCost := sax.QueryCost{}
	setters = append(setters, sax.WithQueryCost(&queryCost))
	res, err := lm.Embed(ctx, text, setters...)
	if queryCostTpuMs != nil {
		*queryCostTpuMs = C.int(queryCost.TpuMs)
	}
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	ret := &lmpb.EmbedResponse{
		Embedding: res,
	}

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_gradient
func go_gradient(ptr C.long, timeout C.float, requestData *C.char, requestSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, queryCostTpuMs *C.int, errMsg **C.char, errCode *C.int) {
	lm := rcgo.Handle(ptr).Value().(*sax.LanguageModel)
	if lm == nil {
		// This is not expected.
		log.Fatalf("gradient() called on nil language model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	requestByte := C.GoBytes(unsafe.Pointer(requestData), requestSize)
	request := &lmpb.GradientRequest{}
	if err := proto.Unmarshal(requestByte, request); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	setters := protoOptionToSetter(options)
	queryCost := sax.QueryCost{}
	setters = append(setters, sax.WithQueryCost(&queryCost))
	score, gradients, err := lm.Gradient(ctx, request.GetPrefix(), request.GetSuffix(), setters...)
	if queryCostTpuMs != nil {
		*queryCostTpuMs = C.int(queryCost.TpuMs)
	}
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	resp := &lmpb.GradientResponse{}
	resp.Score = score
	resp.Gradients = make(map[string]*lmpb.GradientResponse_Gradient, len(gradients))
	for k, v := range gradients {
		resp.Gradients[k] = &lmpb.GradientResponse_Gradient{Values: v}
	}
	content, err := proto.Marshal(resp)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//////////////////////////////////////////////////////////////////////////
// Multimodal model methods
//////////////////////////////////////////////////////////////////////////

//export go_mm_generate
func go_mm_generate(ptr C.long, timeout C.float, requestData *C.char, requestSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	mm := rcgo.Handle(ptr).Value().(*sax.MultimodalModel)
	if mm == nil {
		// This is not expected.
		log.Fatalf("generate() called on nil multimodal model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	requestByte := C.GoBytes(unsafe.Pointer(requestData), requestSize)
	request := &mmpb.GenerateRequest{}
	if err := proto.Unmarshal(requestByte, request); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	resp, err := mm.Generate(ctx, request, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	content, err := proto.Marshal(resp)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_mm_score
func go_mm_score(ptr C.long, timeout C.float, requestData *C.char, requestSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	mm := rcgo.Handle(ptr).Value().(*sax.MultimodalModel)
	if mm == nil {
		// This is not expected.
		log.Fatalf("score() called on nil multimodal model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	requestByte := C.GoBytes(unsafe.Pointer(requestData), requestSize)
	request := &mmpb.ScoreRequest{}
	if err := proto.Unmarshal(requestByte, request); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	resp, err := mm.Score(ctx, request, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	content, err := proto.Marshal(resp)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_mm_embed
func go_mm_embed(ptr C.long, timeout C.float, requestData *C.char, requestSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	mm := rcgo.Handle(ptr).Value().(*sax.MultimodalModel)
	if mm == nil {
		// This is not expected.
		log.Fatalf("embed() called on nil multimodal model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	requestByte := C.GoBytes(unsafe.Pointer(requestData), requestSize)
	request := &mmpb.EmbedRequest{}
	if err := proto.Unmarshal(requestByte, request); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	resp, err := mm.Embed(ctx, request, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	content, err := proto.Marshal(resp)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//////////////////////////////////////////////////////////////////////////
// Vision model methods
//////////////////////////////////////////////////////////////////////////

//export go_classify
func go_classify(ptr C.long, timeout C.float, textData *C.char, textSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("classify() called on nil vision model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	text := C.GoBytes(unsafe.Pointer(textData), textSize)
	res, err := vm.Classify(ctx, text, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	ret := &vmpb.ClassifyResponse{}
	for _, v := range res {
		item := &vmpb.DecodedText{
			Text:  v.Text,
			Score: v.Score,
		}
		ret.Texts = append(ret.GetTexts(), item)
	}

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_text_to_image
func go_text_to_image(ptr C.long, timeout C.float, textData *C.char, textSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("text_to_image() called on nil vision model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	text := C.GoStringN(textData, textSize)
	res, err := vm.TextToImage(ctx, text, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	ret := &vmpb.TextToImageResponse{}
	items := []*vmpb.ImageGenerations{}
	for _, v := range res {
		item := &vmpb.ImageGenerations{
			Image: v.Image,
			Score: v.Logp,
		}
		items = append(items, item)
	}
	ret.Images = items

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_text_and_image_to_image
func go_text_and_image_to_image(ptr C.long, timeout C.float, textData *C.char, textSize C.int, imageData *C.char, imageSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("text_and_image_to_image() called on nil vision model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	text := C.GoStringN(textData, textSize)
	image := C.GoBytes(unsafe.Pointer(imageData), imageSize)
	res, err := vm.TextAndImageToImage(ctx, text, image, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	ret := &vmpb.TextAndImageToImageResponse{}
	items := []*vmpb.ImageGenerations{}
	for _, v := range res {
		item := &vmpb.ImageGenerations{
			Image: v.Image,
			Score: v.Logp,
		}
		items = append(items, item)
	}
	ret.Images = items

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_vm_embed
func go_vm_embed(ptr C.long, timeout C.float, imageData *C.char, imageSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("embed() called on nil vision model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	image := C.GoBytes(unsafe.Pointer(imageData), imageSize)
	res, err := vm.Embed(ctx, image, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	ret := &vmpb.EmbedResponse{
		Embedding: res,
	}

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_vm_detect
func go_vm_detect(ptr C.long, timeout C.float, imageData *C.char, imageSize C.int, reqData *C.char, reqSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("detect() called on nil vision model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	image := C.GoBytes(unsafe.Pointer(imageData), imageSize)

	requestBytes := C.GoStringN(reqData, reqSize)
	request := &vmpb.DetectRequest{}
	if err := proto.Unmarshal([]byte(requestBytes), request); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	var boxes []sax.BoundingBox
	for _, box := range request.GetBoxesOfInterest() {
		boxes = append(boxes, sax.BoundingBox{
			CenterX: box.GetCx(),
			CenterY: box.GetCy(),
			Width:   box.GetW(),
			Height:  box.GetH(),
		})
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}
	res, err := vm.Detect(ctx, []byte(image), request.GetText(), boxes, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	ret := &vmpb.DetectResponse{}
	items := []*vmpb.BoundingBox{}
	for _, v := range res {
		mask := &vmpb.DetectionMask{
			MaskHeight: v.Mask.Height,
			MaskWidth:  v.Mask.Width,
			MaskValues: v.Mask.MaskValues,
		}
		item := &vmpb.BoundingBox{
			Cx:    v.CenterX,
			Cy:    v.CenterY,
			W:     v.Width,
			H:     v.Height,
			Text:  v.Text,
			Score: v.Score,
			Mask:  mask,
		}
		items = append(items, item)
	}
	ret.BoundingBoxes = items

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_vm_image_to_text
func go_vm_image_to_text(ptr C.long, timeout C.float, imageData *C.char, imageSize C.int, textData *C.char, textSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("image_to_text() called on nil vision model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	text := C.GoBytes(unsafe.Pointer(textData), textSize)
	image := C.GoBytes(unsafe.Pointer(imageData), imageSize)
	res, err := vm.ImageToText(ctx, []byte(image), string(text), protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ret := &vmpb.ImageToTextResponse{}
	for _, v := range res {
		item := &vmpb.DecodedText{
			Text:  v.Text,
			Score: v.Score,
		}
		ret.Texts = append(ret.GetTexts(), item)
	}

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_vm_image_to_image
func go_vm_image_to_image(ptr C.long, timeout C.float, imageData *C.char, imageSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("image_to_image() called on nil vision model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	image := C.GoBytes(unsafe.Pointer(imageData), imageSize)
	res, err := vm.ImageToImage(ctx, []byte(image), protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ret := &vmpb.ImageToImageResponse{}
	for _, v := range res {
		item := &vmpb.ImageGenerations{
			Image: v.Image,
			Score: v.Score,
		}
		ret.Images = append(ret.GetImages(), item)
	}

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_vm_video_to_text
func go_vm_video_to_text(ptr C.long, timeout C.float, imageFramesData **C.char,
	perFrameSizes *C.int,
	numFrames C.int,
	textData *C.char, textSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("video_to_text() called on nil vision model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	text := C.GoBytes(unsafe.Pointer(textData), textSize)
	imageFrames := [][]byte{}
	framePtrStart := unsafe.Pointer(imageFramesData)
	frameSizesPtrStart := unsafe.Pointer(perFrameSizes)
	framePtrSize := unsafe.Sizeof(*imageFramesData)
	frameSizesPtrSize := unsafe.Sizeof(*perFrameSizes)

	for i := 0; i < int(numFrames); i++ {
		framePtr := uintptr(framePtrStart) + uintptr(i)*(framePtrSize)
		frameSizePtr := uintptr(frameSizesPtrStart) + uintptr(i)*(frameSizesPtrSize)
		frameSize := *((*C.int)(unsafe.Pointer(frameSizePtr)))
		frame := *((**C.char)(unsafe.Pointer(framePtr)))
		frameBytes := C.GoBytes(unsafe.Pointer(frame), frameSize)
		imageFrames = append(imageFrames, frameBytes)
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	res, err := vm.VideoToText(ctx, imageFrames, string(text), protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ret := &vmpb.VideoToTextResponse{}
	for _, v := range res {
		item := &vmpb.DecodedText{
			Text:  v.Text,
			Score: v.Score,
		}
		ret.Texts = append(ret.GetTexts(), item)
	}

	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_vm_video_to_token
func go_vm_video_to_token(ptr C.long, timeout C.float, imageFramesData **C.char,
	perFrameSizes *C.int,
	numFrames C.int,
	optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("video_to_token() called on nil vision model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	imageFrames := [][]byte{}
	framePtrStart := unsafe.Pointer(imageFramesData)
	frameSizesPtrStart := unsafe.Pointer(perFrameSizes)
	framePtrSize := unsafe.Sizeof(*imageFramesData)
	frameSizesPtrSize := unsafe.Sizeof(*perFrameSizes)
	for i := 0; i < int(numFrames); i++ {
		framePtr := uintptr(framePtrStart) + uintptr(i)*(framePtrSize)
		frameSizePtr := uintptr(frameSizesPtrStart) + uintptr(i)*(frameSizesPtrSize)
		frameSize := *((*C.int)(unsafe.Pointer(frameSizePtr)))
		frame := *((**C.char)(unsafe.Pointer(framePtr)))
		frameBytes := C.GoBytes(unsafe.Pointer(frame), frameSize)
		imageFrames = append(imageFrames, frameBytes)
	}
	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	res, err := vm.VideoToToken(ctx, imageFrames, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ret := &vmpb.VideoToTokenResponse{
		Tokens: res,
	}
	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//export go_vm_token_to_video
func go_vm_token_to_video(ptr C.long, timeout C.float, tokensData *C.double, tokensSize C.int,
	optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int,
	errMsg **C.char, errCode *C.int) {
	vm := rcgo.Handle(ptr).Value().(*sax.VisionModel)
	if vm == nil {
		// This is not expected.
		log.Fatalf("token_to_video() called on nil vision model.")
	}
	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	tokensPtr := unsafe.Pointer(tokensData)
	tokens := unsafe.Slice((*float64)(tokensPtr), tokensSize)

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}
	res, err := vm.TokenToVideo(ctx, tokens, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ret := &vmpb.TokenToVideoResponse{}
	for _, imageBytes := range res {
		ret.ImageFrames = append(ret.GetImageFrames(), imageBytes)
	}
	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)

}

//export go_custom
func go_custom(ptr C.long, timeout C.float, requestData unsafe.Pointer, requestSize C.int, methodNameData *C.char, methodNameSize C.int, optionsData *C.char, optionsSize C.int, outData **C.char, outSize *C.int, errMsg **C.char, errCode *C.int) {
	custom := rcgo.Handle(ptr).Value().(*sax.CustomModel)
	if custom == nil {
		// This is not expected.
		log.Fatalf("custom() called on nil custom model.")
	}

	optionsByte := C.GoBytes(unsafe.Pointer(optionsData), optionsSize)
	options := &cpb.ExtraInputs{}
	if err := proto.Unmarshal(optionsByte, options); err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}

	ctx, cancel := createContextWithTimeout(timeout)
	if cancel != nil {
		defer cancel()
	}

	request := C.GoBytes(requestData, requestSize)
	methodName := C.GoStringN(methodNameData, methodNameSize)
	res, err := custom.Custom(ctx, request, methodName, protoOptionToSetter(options)...)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	ret := &cmpb.CustomResponse{}
	ret.Response = res
	content, err := proto.Marshal(ret)
	if err != nil {
		buildReturnValues(outData, outSize, errMsg, errCode, nil, err)
		return
	}
	buildReturnValues(outData, outSize, errMsg, errCode, &content, nil)
}

//////////////////////////////////////////////////////////////////////////
// Debugging methods
//////////////////////////////////////////////////////////////////////////

//export go_start_debug
func go_start_debug(port C.int) {
	sax.StartDebugPort(context.Background(), int(port))
}

func main() {}
