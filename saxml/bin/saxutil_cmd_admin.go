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

// Package saxcommand contains commands for saxutil.
package saxcommand

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"

	"flag"
	log "github.com/golang/glog"
	"google.golang.org/protobuf/proto"
	"github.com/google/subcommands"
	"github.com/olekukonko/tablewriter"
	"saxml/admin/validator"
	"saxml/client/go/saxadmin"
	"saxml/common/cell"
	"saxml/common/config"
	"saxml/common/naming"
	"saxml/common/watchable"

	apb "saxml/protobuf/admin_go_proto_grpc"
	cpb "saxml/protobuf/common_go_proto"
)

const (
	cmdTimeout = 60 * time.Second
)

// CreateCmd is the command for Create.
type CreateCmd struct{}

// Name returns the name of CreateCmd.
func (*CreateCmd) Name() string { return "create" }

// Synopsis returns the synopsis of CreateCmd.
func (*CreateCmd) Synopsis() string { return "create a SAX cell." }

// Usage returns the full usage of CreateCmd.
func (*CreateCmd) Usage() string {
	return `create <cell name> <file system path> [admin ACL string]:
	Create a SAX cell and initialize its state.
`
}

// SetFlags sets flags for CreateCmd.
func (c *CreateCmd) SetFlags(f *flag.FlagSet) {}

// Execute executes CreateCmd.
func (c *CreateCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 2 && len(f.Args()) != 3 {
		log.Errorf("Provide a SAX cell name (e.g. /sax/bar), a file system path for persistence (e.g. gs://bucket/path), and an optional admin ACL string.")
		return subcommands.ExitUsageError
	}
	saxCell := f.Args()[0]
	fsRoot := f.Args()[1]
	var adminACL string
	if len(f.Args()) == 3 {
		adminACL = f.Args()[2]
	}

	// adminACL, if non-empty, is set as the write ACL on the created cell subdirectory.
	// All files created within will inherit this ACL as writer.
	if err := cell.Create(ctx, saxCell, adminACL); err != nil {
		log.Errorf("Failed to create SAX cell %s: %v", saxCell, err)
		return subcommands.ExitFailure
	}
	if err := config.Create(ctx, saxCell, fsRoot, adminACL); err != nil {
		log.Errorf("Failed to create config %s: %v", saxCell, err)
		return subcommands.ExitFailure
	}

	return subcommands.ExitSuccess
}

func randomSelectAddress(address []string) string {
	n := len(address)
	if n == 0 {
		return ""
	}
	idx := rand.Intn(n)
	return address[idx]
}

// ListCmd is the command for List.
type ListCmd struct{}

// Name returns the name of ListCmd.
func (*ListCmd) Name() string { return "ls" }

// Synopsis returns the synopsis of ListCmd.
func (*ListCmd) Synopsis() string { return "list published models" }

// Usage returns the full usage of ListCmd.
func (*ListCmd) Usage() string {
	return `ls /sax[/cell[/model]]:
	List all sax cells, models in a cell, or information of a model.
`
}

// SetFlags sets flags for ListCmd.
func (c *ListCmd) SetFlags(f *flag.FlagSet) {}

func (c *ListCmd) handleSax(ctx context.Context) subcommands.ExitStatus {
	cells, err := cell.ListAll(ctx)
	if err != nil {
		log.Errorf("Failed to list all cells: %v", err)
		return subcommands.ExitFailure
	}
	sort.Strings(cells)
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"#", "Cell"})
	for idx, cell := range cells {
		table.Append([]string{strconv.Itoa(idx), cell})
	}
	table.Render()
	return subcommands.ExitSuccess
}

func (c *ListCmd) handleSaxCell(ctx context.Context, cellFullName naming.CellFullName) subcommands.ExitStatus {
	admin := saxadmin.Open(cellFullName.CellFullName())

	listResp, err := admin.ListAll(ctx)
	if err != nil {
		log.Errorf("Failed to list models: %v", err)
		return subcommands.ExitFailure
	}
	models := listResp.GetPublishedModels()
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"#", "Model ID"})
	sort.Slice(models, func(i, j int) bool { return models[i].GetModel().GetModelId() < models[j].GetModel().GetModelId() })
	for idx, model := range models {
		table.Append([]string{strconv.Itoa(idx), model.GetModel().GetModelId()[len(cellFullName.CellFullName())+1:]})
	}
	table.Render()
	return subcommands.ExitSuccess
}

func (c *ListCmd) handleSaxModel(ctx context.Context, modelFullName naming.ModelFullName) subcommands.ExitStatus {
	admin := saxadmin.Open(modelFullName.CellFullName())

	publishedModel, err := admin.List(ctx, modelFullName.ModelFullName())
	if err != nil || publishedModel == nil {
		log.Errorf("Failed to list model: %v", err)
		return subcommands.ExitFailure
	}
	model := publishedModel.GetModel()

	// Print out list results in tables.
	// Extra logic: display one random address if there are multiple.
	randomSelectedAddress := randomSelectAddress(publishedModel.GetModeletAddresses())
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Model", "Model Path", "Checkpoint Path", "# of Replicas", "(Selected) ReplicaAddress"})
	table.Append([]string{modelFullName.ModelName(), model.GetModelPath(), model.GetCheckpointPath(), strconv.Itoa(len(publishedModel.GetModeletAddresses())), randomSelectedAddress})
	table.Render()

	table = tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Method", "ACL"})
	aclItems := model.GetAcls().GetItems()
	methods := make([]string, 0, len(aclItems))
	for method := range model.GetAcls().GetItems() {
		methods = append(methods, method)
	}
	sort.Strings(methods)
	for _, method := range methods {
		table.Append([]string{method, aclItems[method]})
	}
	table.Render()
	return subcommands.ExitSuccess
}

// Execute executes ListCmd.
func (c *ListCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 1 {
		log.Errorf("Provide /sax, /sax/<cell>, or /sax/<cell>/<model>")
		return subcommands.ExitUsageError
	}
	arg0 := f.Arg(0)
	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()

	if arg0 == naming.Sax() {
		return c.handleSax(ctx)
	}

	if cellFullName, err := naming.NewCellFullName(arg0); err == nil {
		return c.handleSaxCell(ctx, cellFullName)
	}

	if modelFullName, err := naming.NewModelFullName(arg0); err == nil {
		return c.handleSaxModel(ctx, modelFullName)
	}

	log.Errorf("Invalid model ID %s:", arg0)
	return subcommands.ExitFailure
}

// PublishCmd is the command for Publish.
type PublishCmd struct{}

// Name returns the name of PublishCmd.
func (*PublishCmd) Name() string { return "publish" }

// Synopsis returns the synopsis of PublishCmd.
func (*PublishCmd) Synopsis() string { return "publish a model" }

// Usage returns the full usage of PublishCmd.
func (*PublishCmd) Usage() string {
	return `publish <model ID> <model path> <checkpoint path> <num of replicas>:
	Publish a model using the given number of server replicas.
`
}

// SetFlags sets flags for PublishCmd.
func (c *PublishCmd) SetFlags(f *flag.FlagSet) {}

// Execute executes PublishCmd.
func (c *PublishCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 4 {
		log.Errorf("Provide model ID, model path, checkpoint path, and number of replicas")
		return subcommands.ExitUsageError
	}
	modelID, err := naming.NewModelFullName(f.Args()[0])
	if err != nil {
		log.Errorf("Invalid model ID %s, should be /sax/<cell>/<model>: %v", f.Args()[0], err)
		return subcommands.ExitFailure
	}
	modelPath := f.Args()[1]
	ckptPath := f.Args()[2]
	numReplicas, err := strconv.Atoi(f.Args()[3])
	if err != nil {
		log.Errorf("Provide number of replicas: %v", err)
		return subcommands.ExitUsageError
	}

	admin := saxadmin.Open(modelID.CellFullName())

	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()
	if err := admin.Publish(ctx, modelID.ModelFullName(), modelPath, ckptPath, numReplicas); err != nil {
		log.Errorf("Failed to publish model: %v", err)
		return subcommands.ExitFailure
	}

	return subcommands.ExitSuccess
}

// UnpublishCmd is the command for Unpublish.
type UnpublishCmd struct{}

// Name returns the name of UnpublishCmd.
func (*UnpublishCmd) Name() string { return "unpublish" }

// Synopsis returns the synopsis of UnpublishCmd.
func (*UnpublishCmd) Synopsis() string { return "unpublish a model" }

// Usage returns the full usage of UnpublishCmd.
func (*UnpublishCmd) Usage() string {
	return `unpublish <model ID>:
	Unpublish a published model.
`
}

// SetFlags sets flags for UnpublishCmd.
func (c *UnpublishCmd) SetFlags(f *flag.FlagSet) {}

// Execute executes UnpublishCmd.
func (c *UnpublishCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 1 {
		log.Errorf("Provide a single model ID")
		return subcommands.ExitUsageError
	}
	modelID, err := naming.NewModelFullName(f.Args()[0])
	if err != nil {
		log.Errorf("Invalid model ID %s, should be /sax/<cell>/<model>: %v", f.Args()[0], err)
		return subcommands.ExitFailure
	}

	admin := saxadmin.Open(modelID.CellFullName())

	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()
	if err := admin.Unpublish(ctx, modelID.ModelFullName()); err != nil {
		log.Errorf("Failed to unpublish model: %v", err)
		return subcommands.ExitFailure
	}

	return subcommands.ExitSuccess
}

// UpdateCmd is the command for Update.
type UpdateCmd struct {
	numReplicas int
}

// Name returns the name of UpdateCmd.
func (*UpdateCmd) Name() string { return "update" }

// Synopsis returns the synopsis of UpdateCmd.
func (*UpdateCmd) Synopsis() string { return "update a model" }

// Usage returns the full usage of UpdateCmd.
func (*UpdateCmd) Usage() string {
	return `update [-replicas=<num>]  <model ID>:
	Update a published model.
`
}

// SetFlags sets flags for UpdateCmd.
func (c *UpdateCmd) SetFlags(f *flag.FlagSet) {
	f.IntVar(&c.numReplicas, "replicas", -1, "Number of replicas for this model.")
}

// Execute executes UpdateCmd.
func (c *UpdateCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 1 {
		log.Errorf("Provide a model ID.")
		return subcommands.ExitUsageError
	}
	modelID, err := naming.NewModelFullName(f.Args()[0])
	if err != nil {
		log.Errorf("Invalid model ID %s, should be /sax/<cell>/<model>: %v", f.Args()[0], err)
		return subcommands.ExitFailure
	}

	admin := saxadmin.Open(modelID.CellFullName())

	// Read the current model definition in proto.
	publishedModel, err := admin.List(ctx, modelID.ModelFullName())
	if err != nil || publishedModel == nil {
		log.Errorf("Failed to list model: %v", err)
		return subcommands.ExitFailure
	}
	model := publishedModel.GetModel()
	log.Infof("Current model definition:\n%v", model)

	if c.numReplicas < 0 {
		log.Errorf("Num replicas must be non-negative.")
		return subcommands.ExitFailure
	}
	model.RequestedNumReplicas = int32(c.numReplicas)
	log.Infof("Updated model definition:\n%v", model)

	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()
	if err := admin.Update(ctx, model); err != nil {
		log.Errorf("Failed to update model: %v", err)
		return subcommands.ExitFailure
	}

	return subcommands.ExitSuccess
}

// SetACLCmd is the command for SetACL.
type SetACLCmd struct{}

// Name returns the name of SetACLCmd.
func (*SetACLCmd) Name() string { return "setacl" }

// Synopsis returns the synopsis of SetACLCmd.
func (*SetACLCmd) Synopsis() string { return "set ACLs for a cell or model" }

// Usage returns the full usage of SetACLCmd.
func (*SetACLCmd) Usage() string {
	return `setacl <cell ID> <ACL name> | <model ID> [<method>] <ACL name>:
    Set the ACL of a cell, model, or model's method.
    E.g.,
		$ saxutil setacl /sax/test mdb/admin-group
		$ saxutil setacl /sax/test/lm mdb/admin-group
    $ saxutil setacl /sax/test/lm lm.generate mdb/user-group
    $ saxutil setacl /sax/test/lm lm.score mdb/all
`
}

// SetFlags sets flags for SetACLCmd.
func (c *SetACLCmd) SetFlags(f *flag.FlagSet) {}

func (c *SetACLCmd) handleSaxCell(ctx context.Context, cellFullName naming.CellFullName, args []string) subcommands.ExitStatus {
	saxCell := cellFullName.CellFullName()
	cfg, err := config.Load(ctx, saxCell)
	if err != nil {
		log.Errorf("Failed to load config: %v", err)
		return subcommands.ExitFailure
	}
	log.Infof("Current config definition:\n%v", cfg)

	change := proto.Clone(cfg).(*apb.Config)
	change.AdminAcl = args[1]
	log.Infof("Updated config definition:\n%v", change)

	if err := validator.ValidateConfigUpdate(cfg, change); err != nil {
		log.Errorf("Invalid config update: %v", err)
		return subcommands.ExitFailure
	}
	if err := config.Save(ctx, change, saxCell); err != nil {
		log.Errorf("Failed to save config: %v", err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}

func (c *SetACLCmd) handleSaxModel(ctx context.Context, modelFullName naming.ModelFullName, args []string) subcommands.ExitStatus {
	admin := saxadmin.Open(modelFullName.CellFullName())

	// Read the current model definition in proto.
	publishedModel, err := admin.List(ctx, modelFullName.ModelFullName())
	if err != nil || publishedModel == nil {
		log.Errorf("Failed to list model: %v", err)
		return subcommands.ExitFailure
	}
	model := publishedModel.GetModel()
	log.Infof("Current model definition:\n%v", model)

	// Set model admin method ACLs.
	if len(args) == 2 {
		model.AdminAcl = args[1]

		log.Infof("Updated model definition:\n%v", model)
		if err := admin.Update(ctx, model); err != nil {
			log.Errorf("Failed to update model: %v", err)
			return subcommands.ExitFailure
		}
		return subcommands.ExitSuccess
	}

	// Set model data method ACLs.
	acls := model.GetAcls()
	if acls == nil {
		acls = &cpb.AccessControlLists{}
		model.Acls = acls
	}
	items := acls.GetItems()
	if items == nil {
		items = make(map[string]string)
		acls.Items = items
	}
	method := args[1]
	aclname := args[2]
	if aclname == "" {
		delete(items, method)
	} else {
		items[method] = aclname
	}

	log.Infof("Updated model definition:\n%v", model)
	if err := admin.Update(ctx, model); err != nil {
		log.Errorf("Failed to update model: %v", err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}

// Execute executes SetACLCmd.
func (c *SetACLCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 2 && len(f.Args()) != 3 {
		log.Errorf("Provide a cell or model ID, optionally a model method, and an ACL.")
		return subcommands.ExitUsageError
	}
	arg0 := f.Arg(0)
	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()

	if cellFullName, err := naming.NewCellFullName(arg0); err == nil {
		return c.handleSaxCell(ctx, cellFullName, f.Args())
	}

	if modelFullName, err := naming.NewModelFullName(arg0); err == nil {
		return c.handleSaxModel(ctx, modelFullName, f.Args())
	}

	log.Errorf("Invalid cell or model ID: %s", arg0)
	return subcommands.ExitFailure
}

// GetACLCmd is the command for GetACL.
type GetACLCmd struct{}

// Name returns the name of GetACLCmd.
func (*GetACLCmd) Name() string { return "getacl" }

// Synopsis returns the synopsis of GetACLCmd.
func (*GetACLCmd) Synopsis() string { return "get ACLs for a cell or model" }

// Usage returns the full usage of GetACLCmd.
func (*GetACLCmd) Usage() string {
	return `getacl <cell ID> | <model ID> [all | <method>]:
    Get the ACL of a cell, model, or model's method.
    E.g.,
		$ saxutil getacl /sax/test
    $ saxutil getacl /sax/test/lm
    $ saxutil getacl /sax/test/lm all
    $ saxutil getacl /sax/test/lm lm.generate
`
}

// SetFlags sets flags for GetACLCmd.
func (c *GetACLCmd) SetFlags(f *flag.FlagSet) {}

func (c *GetACLCmd) handleSaxCell(ctx context.Context, cellFullName naming.CellFullName, args []string) subcommands.ExitStatus {
	saxCell := cellFullName.CellFullName()
	cfg, err := config.Load(ctx, saxCell)
	if err != nil {
		log.Errorf("Failed to load config: %v", err)
		return subcommands.ExitFailure
	}
	log.Infof("Current config definition:\n%v", cfg)

	if cfg.GetAdminAcl() == "" {
		fmt.Println("No admin ACLs set for cell.")
	} else {
		fmt.Println(cfg.GetAdminAcl())
	}
	return subcommands.ExitSuccess
}

func (c *GetACLCmd) handleSaxModel(ctx context.Context, modelFullName naming.ModelFullName, args []string) subcommands.ExitStatus {
	admin := saxadmin.Open(modelFullName.CellFullName())

	// Read the current model definition in proto.
	publishedModel, err := admin.List(ctx, modelFullName.ModelFullName())
	if err != nil || publishedModel == nil {
		log.Errorf("Failed to list model: %v", err)
		return subcommands.ExitFailure
	}
	model := publishedModel.GetModel()
	log.Infof("Current model definition:\n%v", model)

	// Get model admin method ACLs.
	if len(args) == 1 {
		acl := model.GetAdminAcl()
		if acl == "" {
			fmt.Println("No admin ACLs set for model.")
		} else {
			fmt.Println(acl)
		}
		return subcommands.ExitSuccess
	}

	// Get model data method ACLs.
	acls := model.GetAcls()
	if acls == nil {
		fmt.Println("No ACLs set for model.")
		return subcommands.ExitSuccess
	}
	items := acls.GetItems()
	if items == nil {
		fmt.Println("No ACLs set for model.")
		return subcommands.ExitSuccess
	}
	method := args[1]
	if method == "all" {
		for k, v := range items {
			fmt.Printf("%s: %s\n", k, v)
		}
	} else {
		acl, ok := items[method]
		if !ok {
			fmt.Println("No ACLs set for method.")
		} else {
			fmt.Println(acl)
		}
	}
	return subcommands.ExitSuccess
}

// Execute executes GetACLCmd.
func (c *GetACLCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 1 && len(f.Args()) != 2 {
		log.Errorf("Provide a cell or model ID, and optionally a method.")
		return subcommands.ExitUsageError
	}
	arg0 := f.Arg(0)
	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()

	if cellFullName, err := naming.NewCellFullName(arg0); err == nil {
		return c.handleSaxCell(ctx, cellFullName, f.Args())
	}

	if modelFullName, err := naming.NewModelFullName(arg0); err == nil {
		return c.handleSaxModel(ctx, modelFullName, f.Args())
	}

	log.Errorf("Invalid cell or model ID: %s", arg0)
	return subcommands.ExitFailure
}

// WatchCmd is the command for WatchAddresses. Useful for debugging.
type WatchCmd struct{}

// Name returns the name of WatchCmd.
func (*WatchCmd) Name() string { return "watch" }

// Synopsis returns the synopsis of WatchCmd.
func (*WatchCmd) Synopsis() string { return "watch addresses of a model" }

// Usage returns the full usage of WatchCmd.
func (*WatchCmd) Usage() string {
	return `watch <model ID>:
    Watch server addresses of a model.
`
}

// SetFlags sets flags for WatchCmd.
func (c *WatchCmd) SetFlags(f *flag.FlagSet) {}

// Execute executes WatchCmd.
func (c *WatchCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 1 {
		log.Errorf("Provide a model ID.")
		return subcommands.ExitUsageError
	}
	modelID, err := naming.NewModelFullName(f.Args()[0])
	if err != nil {
		log.Errorf("Invalid model ID %s, should be /sax/<cell>/<model>: %v", f.Args()[0], err)
		return subcommands.ExitFailure
	}

	admin := saxadmin.Open(modelID.CellFullName())
	ch := make(chan *saxadmin.WatchResult)
	go admin.WatchAddresses(ctx, modelID.ModelFullName(), ch)
	for {
		wr := <-ch
		if wr.Err != nil {
			log.Errorf("WatchAddresses(%v) error: %v", modelID.ModelFullName(), wr.Err)
			return subcommands.ExitFailure
		}
		if wr.Result.Data != nil {
			addrs := wr.Result.Data.ToList()
			sort.Strings(addrs)
			fmt.Println("Reset")
			for _, addr := range addrs {
				fmt.Printf("+ %v\n", addr)
			}
		}
		for _, m := range wr.Result.Log {
			switch m.Kind {
			case watchable.Add:
				fmt.Printf("+ %v\n", m.Val)
			case watchable.Del:
				fmt.Printf("- %v\n", m.Val)
			default:
				log.Warningf("Unexpected Kind: %v", m.Kind)
			}
		}
	}
}
