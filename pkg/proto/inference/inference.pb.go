// Code generated for InferenceService proto definitions. DO NOT EDIT directly unless updating proto.
package inference

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ModelState enumerates the lifecycle states a model can be in.
type ModelState int32

const (
	ModelState_MODEL_STATE_UNKNOWN     ModelState = 0
	ModelState_MODEL_STATE_READY       ModelState = 1
	ModelState_MODEL_STATE_LOADING     ModelState = 2
	ModelState_MODEL_STATE_UNLOADING   ModelState = 3
	ModelState_MODEL_STATE_UNAVAILABLE ModelState = 4
)

// ServingStatus indicates overall readiness of the router.
type ServingStatus int32

const (
	ServingStatus_UNKNOWN     ServingStatus = 0
	ServingStatus_SERVING     ServingStatus = 1
	ServingStatus_NOT_SERVING ServingStatus = 2
)

// TensorData is the wire representation of a named tensor payload.
type TensorData struct {
	Name     string   `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Datatype string   `protobuf:"bytes,2,opt,name=datatype,proto3" json:"datatype,omitempty"`
	Shape    []int64  `protobuf:"varint,3,rep,packed,name=shape,proto3" json:"shape,omitempty"`
	RawData  []byte   `protobuf:"bytes,4,opt,name=raw_data,json=rawData,proto3" json:"raw_data,omitempty"`
}

func (m *TensorData) Reset()         { *m = TensorData{} }
func (m *TensorData) String() string { return m.Name }
func (*TensorData) ProtoMessage()    {}

func (m *TensorData) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}
func (m *TensorData) GetDatatype() string {
	if m != nil {
		return m.Datatype
	}
	return ""
}
func (m *TensorData) GetShape() []int64 {
	if m != nil {
		return m.Shape
	}
	return nil
}
func (m *TensorData) GetRawData() []byte {
	if m != nil {
		return m.RawData
	}
	return nil
}

// InferenceRequest carries prediction inputs for a named model.
type InferenceRequest struct {
	ModelName    string        `protobuf:"bytes,1,opt,name=model_name,json=modelName,proto3" json:"model_name,omitempty"`
	ModelVersion string        `protobuf:"bytes,2,opt,name=model_version,json=modelVersion,proto3" json:"model_version,omitempty"`
	Inputs       []*TensorData `protobuf:"bytes,3,rep,name=inputs,proto3" json:"inputs,omitempty"`
	Priority     int32         `protobuf:"varint,4,opt,name=priority,proto3" json:"priority,omitempty"`
	DeadlineMs   int64         `protobuf:"varint,5,opt,name=deadline_ms,json=deadlineMs,proto3" json:"deadline_ms,omitempty"`
	RequestId    string        `protobuf:"bytes,6,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
}

func (m *InferenceRequest) Reset()         { *m = InferenceRequest{} }
func (m *InferenceRequest) String() string { return m.ModelName }
func (*InferenceRequest) ProtoMessage()    {}

func (m *InferenceRequest) GetModelName() string {
	if m != nil {
		return m.ModelName
	}
	return ""
}
func (m *InferenceRequest) GetModelVersion() string {
	if m != nil {
		return m.ModelVersion
	}
	return ""
}
func (m *InferenceRequest) GetInputs() []*TensorData {
	if m != nil {
		return m.Inputs
	}
	return nil
}
func (m *InferenceRequest) GetPriority() int32 {
	if m != nil {
		return m.Priority
	}
	return 0
}
func (m *InferenceRequest) GetDeadlineMs() int64 {
	if m != nil {
		return m.DeadlineMs
	}
	return 0
}
func (m *InferenceRequest) GetRequestId() string {
	if m != nil {
		return m.RequestId
	}
	return ""
}

// InferenceResponse wraps output tensors and latency telemetry.
type InferenceResponse struct {
	Outputs      []*TensorData `protobuf:"bytes,1,rep,name=outputs,proto3" json:"outputs,omitempty"`
	ModelName    string        `protobuf:"bytes,2,opt,name=model_name,json=modelName,proto3" json:"model_name,omitempty"`
	ModelVersion string        `protobuf:"bytes,3,opt,name=model_version,json=modelVersion,proto3" json:"model_version,omitempty"`
	LatencyMs    float32       `protobuf:"fixed32,4,opt,name=latency_ms,json=latencyMs,proto3" json:"latency_ms,omitempty"`
	RequestId    string        `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
}

func (m *InferenceResponse) Reset()         { *m = InferenceResponse{} }
func (m *InferenceResponse) String() string { return m.ModelName }
func (*InferenceResponse) ProtoMessage()    {}

func (m *InferenceResponse) GetOutputs() []*TensorData {
	if m != nil {
		return m.Outputs
	}
	return nil
}
func (m *InferenceResponse) GetModelName() string {
	if m != nil {
		return m.ModelName
	}
	return ""
}
func (m *InferenceResponse) GetModelVersion() string {
	if m != nil {
		return m.ModelVersion
	}
	return ""
}
func (m *InferenceResponse) GetLatencyMs() float32 {
	if m != nil {
		return m.LatencyMs
	}
	return 0
}
func (m *InferenceResponse) GetRequestId() string {
	if m != nil {
		return m.RequestId
	}
	return ""
}

// ModelStatusRequest queries status for a model.
type ModelStatusRequest struct {
	ModelName    string `protobuf:"bytes,1,opt,name=model_name,json=modelName,proto3" json:"model_name,omitempty"`
	ModelVersion string `protobuf:"bytes,2,opt,name=model_version,json=modelVersion,proto3" json:"model_version,omitempty"`
}

func (m *ModelStatusRequest) Reset()         { *m = ModelStatusRequest{} }
func (m *ModelStatusRequest) String() string { return m.ModelName }
func (*ModelStatusRequest) ProtoMessage()    {}

// BackendModelStatus reports status on a single backend instance.
type BackendModelStatus struct {
	BackendAddress string     `protobuf:"bytes,1,opt,name=backend_address,json=backendAddress,proto3" json:"backend_address,omitempty"`
	State          ModelState `protobuf:"varint,2,opt,name=state,proto3,enum=inference.ModelState" json:"state,omitempty"`
	Reason         string     `protobuf:"bytes,3,opt,name=reason,proto3" json:"reason,omitempty"`
}

func (m *BackendModelStatus) Reset()         { *m = BackendModelStatus{} }
func (m *BackendModelStatus) String() string { return m.BackendAddress }
func (*BackendModelStatus) ProtoMessage()    {}

// ModelStatusResponse carries model status across backends.
type ModelStatusResponse struct {
	ModelName    string                `protobuf:"bytes,1,opt,name=model_name,json=modelName,proto3" json:"model_name,omitempty"`
	ModelVersion string                `protobuf:"bytes,2,opt,name=model_version,json=modelVersion,proto3" json:"model_version,omitempty"`
	State        ModelState            `protobuf:"varint,3,opt,name=state,proto3,enum=inference.ModelState" json:"state,omitempty"`
	Backends     []*BackendModelStatus `protobuf:"bytes,4,rep,name=backends,proto3" json:"backends,omitempty"`
}

func (m *ModelStatusResponse) Reset()         { *m = ModelStatusResponse{} }
func (m *ModelStatusResponse) String() string { return m.ModelName }
func (*ModelStatusResponse) ProtoMessage()    {}

// HealthRequest is an empty probe message.
type HealthRequest struct{}

func (m *HealthRequest) Reset()         { *m = HealthRequest{} }
func (m *HealthRequest) String() string { return "HealthRequest" }
func (*HealthRequest) ProtoMessage()    {}

// HealthResponse reports overall router health and backend counts.
type HealthResponse struct {
	Status          ServingStatus `protobuf:"varint,1,opt,name=status,proto3,enum=inference.ServingStatus" json:"status,omitempty"`
	HealthyBackends int32         `protobuf:"varint,2,opt,name=healthy_backends,json=healthyBackends,proto3" json:"healthy_backends,omitempty"`
	TotalBackends   int32         `protobuf:"varint,3,opt,name=total_backends,json=totalBackends,proto3" json:"total_backends,omitempty"`
	Message         string        `protobuf:"bytes,4,opt,name=message,proto3" json:"message,omitempty"`
}

func (m *HealthResponse) Reset()         { *m = HealthResponse{} }
func (m *HealthResponse) String() string { return m.Message }
func (*HealthResponse) ProtoMessage()    {}

// InferenceServiceClient is the client API for InferenceService.
type InferenceServiceClient interface {
	Predict(ctx context.Context, in *InferenceRequest, opts ...grpc.CallOption) (*InferenceResponse, error)
	ModelStatus(ctx context.Context, in *ModelStatusRequest, opts ...grpc.CallOption) (*ModelStatusResponse, error)
	Health(ctx context.Context, in *HealthRequest, opts ...grpc.CallOption) (*HealthResponse, error)
}

type inferenceServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewInferenceServiceClient(cc grpc.ClientConnInterface) InferenceServiceClient {
	return &inferenceServiceClient{cc}
}

func (c *inferenceServiceClient) Predict(ctx context.Context, in *InferenceRequest, opts ...grpc.CallOption) (*InferenceResponse, error) {
	out := new(InferenceResponse)
	err := c.cc.Invoke(ctx, "/inference.InferenceService/Predict", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *inferenceServiceClient) ModelStatus(ctx context.Context, in *ModelStatusRequest, opts ...grpc.CallOption) (*ModelStatusResponse, error) {
	out := new(ModelStatusResponse)
	err := c.cc.Invoke(ctx, "/inference.InferenceService/ModelStatus", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *inferenceServiceClient) Health(ctx context.Context, in *HealthRequest, opts ...grpc.CallOption) (*HealthResponse, error) {
	out := new(HealthResponse)
	err := c.cc.Invoke(ctx, "/inference.InferenceService/Health", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// InferenceServiceServer is the server API for InferenceService.
type InferenceServiceServer interface {
	Predict(context.Context, *InferenceRequest) (*InferenceResponse, error)
	ModelStatus(context.Context, *ModelStatusRequest) (*ModelStatusResponse, error)
	Health(context.Context, *HealthRequest) (*HealthResponse, error)
	mustEmbedUnimplementedInferenceServiceServer()
}

type UnimplementedInferenceServiceServer struct{}

func (UnimplementedInferenceServiceServer) Predict(context.Context, *InferenceRequest) (*InferenceResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Predict not implemented")
}
func (UnimplementedInferenceServiceServer) ModelStatus(context.Context, *ModelStatusRequest) (*ModelStatusResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ModelStatus not implemented")
}
func (UnimplementedInferenceServiceServer) Health(context.Context, *HealthRequest) (*HealthResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Health not implemented")
}
func (UnimplementedInferenceServiceServer) mustEmbedUnimplementedInferenceServiceServer() {}

func RegisterInferenceServiceServer(s grpc.ServiceRegistrar, srv InferenceServiceServer) {
	s.RegisterService(&InferenceService_ServiceDesc, srv)
}

var InferenceService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "inference.InferenceService",
	HandlerType: (*InferenceServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Predict",
			Handler: func(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
				in := new(InferenceRequest)
				if err := dec(in); err != nil {
					return nil, err
				}
				if interceptor == nil {
					return srv.(InferenceServiceServer).Predict(ctx, in)
				}
				info := &grpc.UnaryServerInfo{
					Server:     srv,
					FullMethod: "/inference.InferenceService/Predict",
				}
				handler := func(ctx context.Context, req interface{}) (interface{}, error) {
					return srv.(InferenceServiceServer).Predict(ctx, req.(*InferenceRequest))
				}
				return interceptor(ctx, in, info, handler)
			},
		},
		{
			MethodName: "ModelStatus",
			Handler: func(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
				in := new(ModelStatusRequest)
				if err := dec(in); err != nil {
					return nil, err
				}
				if interceptor == nil {
					return srv.(InferenceServiceServer).ModelStatus(ctx, in)
				}
				info := &grpc.UnaryServerInfo{
					Server:     srv,
					FullMethod: "/inference.InferenceService/ModelStatus",
				}
				handler := func(ctx context.Context, req interface{}) (interface{}, error) {
					return srv.(InferenceServiceServer).ModelStatus(ctx, req.(*ModelStatusRequest))
				}
				return interceptor(ctx, in, info, handler)
			},
		},
		{
			MethodName: "Health",
			Handler: func(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
				in := new(HealthRequest)
				if err := dec(in); err != nil {
					return nil, err
				}
				if interceptor == nil {
					return srv.(InferenceServiceServer).Health(ctx, in)
				}
				info := &grpc.UnaryServerInfo{
					Server:     srv,
					FullMethod: "/inference.InferenceService/Health",
				}
				handler := func(ctx context.Context, req interface{}) (interface{}, error) {
					return srv.(InferenceServiceServer).Health(ctx, req.(*HealthRequest))
				}
				return interceptor(ctx, in, info, handler)
			},
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "pkg/proto/inference.proto",
}

