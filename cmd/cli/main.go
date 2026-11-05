package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"text/tabwriter"
	"time"

	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const version = "1.0.0"

func printUsage() {
	fmt.Println("Distributed GPU Inference Platform CLI (dist-inf)")
	fmt.Println("=================================================")
	fmt.Println("Usage: dist-inf <command> [flags]")
	fmt.Println("")
	fmt.Println("Commands:")
	fmt.Println("  status      Query health and telemetry across all 4 GPU cluster pools")
	fmt.Println("  models      List all loaded TensorRT and PyTorch models")
	fmt.Println("  predict     Submit a test prediction payload to the router")
	fmt.Println("  version     Print CLI version")
	fmt.Println("")
	fmt.Println("Flags:")
	fmt.Println("  -addr       Router gRPC address (default: localhost:50051)")
	fmt.Println("  -model      Model name for predict/models commands")
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]

	fs := flag.NewFlagSet(command, flag.ExitOnError)
	addr := fs.String("addr", "localhost:50051", "Router gRPC server address")
	modelName := fs.String("model", "resnet50", "Target model name")
	priority := fs.Int("priority", 5, "Inference priority (1-10)")

	_ = fs.Parse(os.Args[2:])

	switch command {
	case "version":
		fmt.Printf("dist-inf CLI version %s (Linux/Darwin x86_64/arm64)\n", version)

	case "status":
		queryStatus(*addr)

	case "models":
		queryModels(*addr, *modelName)

	case "predict":
		runPredict(*addr, *modelName, int32(*priority))

	case "help":
		printUsage()

	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", command)
		printUsage()
		os.Exit(1)
	}
}

func connect(addr string) (pb.InferenceServiceClient, *grpc.ClientConn, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(
		ctx,
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("could not connect to router at %s: %w", addr, err)
	}
	return pb.NewInferenceServiceClient(conn), conn, nil
}

func queryStatus(addr string) {
	fmt.Printf("Querying cluster status from router at %s...\n\n", addr)
	client, conn, err := connect(addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Println("Tip: Ensure the Go router is running with: make run")
		os.Exit(1)
	}
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	hResp, err := client.Health(ctx, &pb.HealthRequest{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Health query failed: %v\n", err)
		os.Exit(1)
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', tabwriter.TabIndent)
	fmt.Fprintln(w, "STATUS\tHEALTHY BACKENDS\tTOTAL BACKENDS\tMESSAGE")
	fmt.Fprintf(w, "%v\t%d\t%d\t%s\n", hResp.Status, hResp.HealthyBackends, hResp.TotalBackends, hResp.Message)
	w.Flush()
	fmt.Println("")
}

func queryModels(addr, model string) {
	client, conn, err := connect(addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	resp, err := client.ModelStatus(ctx, &pb.ModelStatusRequest{ModelName: model})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Model query failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Model: %s (State: %v)\n", resp.ModelName, resp.State)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', tabwriter.TabIndent)
	fmt.Fprintln(w, "BACKEND ADDRESS\tSTATE\tREASON")
	for _, b := range resp.Backends {
		fmt.Fprintf(w, "%s\t%v\t%s\n", b.BackendAddress, b.State, b.Reason)
	}
	w.Flush()
}

func runPredict(addr, model string, priority int32) {
	client, conn, err := connect(addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.InferenceRequest{
		ModelName:    model,
		ModelVersion: "1",
		Priority:     priority,
		Inputs: []*pb.TensorData{
			{
				Name:     "input_0",
				Datatype: "FP32",
				Shape:    []int64{1, 3, 224, 224},
				RawData:  make([]byte, 1*3*224*224*4), // Synthetic zero tensor
			},
		},
	}

	t0 := time.Now()
	resp, err := client.Predict(ctx, req)
	elapsed := time.Since(t0)

	if err != nil {
		fmt.Fprintf(os.Stderr, "Inference prediction failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("=================================================")
	fmt.Println(" INFERENCE EXECUTION SUCCESSFUL")
	fmt.Println("=================================================")
	fmt.Printf(" Model Name:      %s (v%s)\n", resp.ModelName, resp.ModelVersion)
	fmt.Printf(" Output Tensors:  %d\n", len(resp.Outputs))
	fmt.Printf(" Server Latency:  %.2f ms\n", resp.LatencyMs)
	fmt.Printf(" Client E2E:      %.2f ms\n", float64(elapsed.Microseconds())/1000.0)
	fmt.Println("=================================================")
}

