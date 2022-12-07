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

// Package config provides functions to create, save, and load server configurations.
package config

import (
	"context"
	"path/filepath"

	log "github.com/golang/glog"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"saxml/common/cell"
	"saxml/common/platform/env"
	pb "saxml/protobuf/admin_go_proto_grpc"
)

const (
	// configFile is the name of the file storing admin configuration such as fsroot.
	configFile = "config.proto"
)

func configFileName(ctx context.Context, saxCell string) (string, error) {
	if err := cell.Exists(ctx, saxCell); err != nil {
		return "", err
	}
	path, err := cell.Path(ctx, saxCell)
	if err != nil {
		return "", err
	}
	return filepath.Join(path, configFile), nil
}

// Load loads the server config.
func Load(ctx context.Context, saxCell string) (*pb.Config, error) {
	fname, err := configFileName(ctx, saxCell)
	if err != nil {
		return nil, err
	}

	out, err := env.Get().ReadFile(ctx, fname)
	if err != nil {
		return nil, err
	}
	config := &pb.Config{}
	if err := proto.Unmarshal(out, config); err != nil {
		return nil, err
	}
	return config, nil
}

// Watch sends server config updates on a channel.
func Watch(ctx context.Context, saxCell string) (<-chan *pb.Config, error) {
	fname, err := configFileName(ctx, saxCell)
	if err != nil {
		return nil, err
	}

	contentUpdates, err := env.Get().Watch(ctx, fname)
	if err != nil {
		return nil, err
	}
	configUpdates := make(chan *pb.Config)
	go func() {
		for content := range contentUpdates {
			config := &pb.Config{}
			if err := proto.Unmarshal(content, config); err != nil {
				log.Errorf("Watch erroro: %v", err)
			} else {
				configUpdates <- config
			}
		}
	}()
	return configUpdates, nil
}

// Save saves the server config.
func Save(ctx context.Context, config *pb.Config, saxCell string) error {
	fname, err := configFileName(ctx, saxCell)
	if err != nil {
		return err
	}

	in, err := proto.Marshal(config)
	if err != nil {
		return err
	}
	return env.Get().WriteFile(ctx, fname, in)
}

// Create creates a server config.
func Create(ctx context.Context, saxCell, fsRoot, adminACL string) error {
	if err := cell.Exists(ctx, saxCell); err != nil {
		return err
	}
	config := &pb.Config{}
	config.FsRoot = fsRoot
	config.AdminAcl = adminACL
	log.Infof("Creating config %v", prototext.Format(config))
	if err := Save(ctx, config, saxCell); err != nil {
		return err
	}
	log.Infof("Created config %v", prototext.Format(config))
	return nil
}
