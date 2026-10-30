// Command gateway is an Envoy ext_proc service that routes LLM requests to the local SLM or cloud
// cluster by setting x-kinetix-* headers (Envoy then selects the cluster from x-kinetix-engine).
// Requires processing_mode request_body_mode: BUFFERED.
package main

import (
	"encoding/json"
	"flag"
	"io"
	"log"
	"net"
	"strconv"
	"strings"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extproc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"

	"kinetix/gateway/routing"
)

type server struct {
	extproc.UnimplementedExternalProcessorServer
	gate routing.Gate
}

type chatBody struct {
	Prompt   string `json:"prompt"`
	Messages []struct {
		Content string `json:"content"`
	} `json:"messages"`
}

func queryText(body []byte) string {
	var b chatBody
	if json.Unmarshal(body, &b) != nil {
		return ""
	}
	parts := []string{b.Prompt}
	for _, m := range b.Messages {
		parts = append(parts, m.Content)
	}
	return strings.Join(parts, " ")
}

func hdrFloat(h *corev3.HeaderMap, key string, def float64) float64 {
	if h == nil {
		return def
	}
	for _, kv := range h.Headers {
		if strings.EqualFold(kv.Key, key) {
			v := string(kv.RawValue)
			if v == "" {
				v = kv.Value
			}
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				return f
			}
		}
	}
	return def
}

func setHeader(k, v string) *corev3.HeaderValueOption {
	return &corev3.HeaderValueOption{Header: &corev3.HeaderValue{Key: k, RawValue: []byte(v)}}
}

func (s *server) Process(stream extproc.ExternalProcessor_ProcessServer) error {
	var uncertainty, risk, budget = 0.1, 0.0, 1.0
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		resp := &extproc.ProcessingResponse{}
		switch v := req.Request.(type) {
		case *extproc.ProcessingRequest_RequestHeaders:
			h := v.RequestHeaders.GetHeaders()
			uncertainty = hdrFloat(h, "x-kinetix-uncertainty", uncertainty)
			risk = hdrFloat(h, "x-kinetix-risk", risk)
			budget = hdrFloat(h, "x-kinetix-budget", budget)
			resp.Response = &extproc.ProcessingResponse_RequestHeaders{RequestHeaders: &extproc.HeadersResponse{}}
		case *extproc.ProcessingRequest_RequestBody:
			d := s.gate.Route(queryText(v.RequestBody.Body), uncertainty, risk, budget)
			resp.Response = &extproc.ProcessingResponse_RequestBody{RequestBody: &extproc.BodyResponse{
				Response: &extproc.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extproc.HeaderMutation{SetHeaders: []*corev3.HeaderValueOption{
						setHeader("x-kinetix-route", d.Route),
						setHeader("x-kinetix-engine", d.Engine),
					}},
				},
			}}
		default:
			continue
		}
		if err := stream.Send(resp); err != nil {
			return err
		}
	}
}

func main() {
	addr := flag.String("addr", ":50051", "gRPC listen address")
	flag.Parse()
	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatal(err)
	}
	g := grpc.NewServer()
	extproc.RegisterExternalProcessorServer(g, &server{gate: routing.DefaultGate()})
	log.Printf("ext_proc listening on %s", *addr)
	log.Fatal(g.Serve(lis))
}
