package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"

	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
)

// CacheEntry stores an inference response with creation timestamp and expiration.
type CacheEntry struct {
	Response  *pb.InferenceResponse
	ExpiresAt time.Time
}

// TensorCache provides in-memory LRU caching with tensor fingerprinting.
type TensorCache struct {
	mu      sync.RWMutex
	entries map[string]CacheEntry
	ttl     time.Duration
	hits    uint64
	misses  uint64
}

// NewTensorCache initializes a local in-memory tensor cache.
func NewTensorCache(ttl time.Duration) *TensorCache {
	c := &TensorCache{
		entries: make(map[string]CacheEntry),
		ttl:     ttl,
	}
	go c.cleanupLoop()
	return c
}

// ComputeFingerprint hashes the model name, version, and raw bytes of all input tensors.
func ComputeFingerprint(req *pb.InferenceRequest) string {
	hasher := sha256.New()
	hasher.Write([]byte(req.ModelName))
	hasher.Write([]byte(":"))
	hasher.Write([]byte(req.ModelVersion))

	for _, in := range req.Inputs {
		hasher.Write([]byte(in.Name))
		hasher.Write([]byte(in.Datatype))
		hasher.Write(in.RawData)
	}

	return hex.EncodeToString(hasher.Sum(nil))
}

// Get checks if an identical inference request has a valid cached result.
func (c *TensorCache) Get(req *pb.InferenceRequest) (*pb.InferenceResponse, bool) {
	key := ComputeFingerprint(req)

	c.mu.RLock()
	entry, exists := c.entries[key]
	c.mu.RUnlock()

	if !exists {
		c.mu.Lock()
		c.misses++
		c.mu.Unlock()
		return nil, false
	}

	if time.Now().After(entry.ExpiresAt) {
		c.mu.Lock()
		delete(c.entries, key)
		c.misses++
		c.mu.Unlock()
		return nil, false
	}

	c.mu.Lock()
	c.hits++
	c.mu.Unlock()

	return entry.Response, true
}

// Set stores a computed response in cache.
func (c *TensorCache) Set(req *pb.InferenceRequest, resp *pb.InferenceResponse) {
	key := ComputeFingerprint(req)

	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries[key] = CacheEntry{
		Response:  resp,
		ExpiresAt: time.Now().Add(c.ttl),
	}
}

// Stats returns hit and miss counts.
func (c *TensorCache) Stats() (hits, misses uint64, ratio float64) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	total := c.hits + c.misses
	r := 0.0
	if total > 0 {
		r = float64(c.hits) / float64(total) * 100.0
	}
	return c.hits, c.misses, r
}

func (c *TensorCache) cleanupLoop() {
	ticker := time.NewTicker(30 * time.Second)
	for now := range ticker.C {
		c.mu.Lock()
		for k, v := range c.entries {
			if now.After(v.ExpiresAt) {
				delete(c.entries, k)
			}
		}
		c.mu.Unlock()
	}
}

