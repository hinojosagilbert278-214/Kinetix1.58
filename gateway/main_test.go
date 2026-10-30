package main

import (
	"context"
	"net"
	"testing"

	extproc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"kinetix/gateway/routing"
)

func TestExtProcRouting(t *testing.T) {
	lis, _ := net.Listen("tcp", "127.0.0.1:0")
	g := grpc.NewServer()
	extproc.RegisterExternalProcessorServer(g, &server{gate: routing.DefaultGate()})
	go g.Serve(lis)
	defer g.Stop()

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	stream, err := extproc.NewExternalProcessorClient(conn).Process(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	stream.Send(&extproc.ProcessingRequest{Request: &extproc.ProcessingRequest_RequestHeaders{RequestHeaders: &extproc.HttpHeaders{}}})
	if _, err := stream.Recv(); err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"messages":[{"content":"summarize my uploaded policy"}]}`)
	stream.Send(&extproc.ProcessingRequest{Request: &extproc.ProcessingRequest_RequestBody{RequestBody: &extproc.HttpBody{Body: body}}})
	resp, err := stream.Recv()
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, h := range resp.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() {
		got[h.Header.Key] = string(h.Header.RawValue)
	}
	if got["x-kinetix-route"] != "private" || got["x-kinetix-engine"] != "local" {
		t.Fatalf("unexpected headers: %v", got)
	}
}
