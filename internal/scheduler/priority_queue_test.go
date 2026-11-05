package scheduler

import (
	"errors"
	"sync"
	"testing"
	"time"

	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
)

func TestPriorityQueueOrdering(t *testing.T) {
	pq := NewPriorityQueue()

	now := time.Now()
	// Push items with varying priorities
	pq.Push(&QueueItem{
		Request:      &pb.InferenceRequest{ModelName: "low-prio"},
		Priority:     1,
		EnqueuedAt:   now,
		ResponseChan: make(chan QueuedResponse, 1),
	})

	pq.Push(&QueueItem{
		Request:      &pb.InferenceRequest{ModelName: "high-prio"},
		Priority:     10,
		EnqueuedAt:   now.Add(1 * time.Millisecond),
		ResponseChan: make(chan QueuedResponse, 1),
	})

	pq.Push(&QueueItem{
		Request:      &pb.InferenceRequest{ModelName: "med-prio"},
		Priority:     5,
		EnqueuedAt:   now.Add(2 * time.Millisecond),
		ResponseChan: make(chan QueuedResponse, 1),
	})

	// Pop 1: High priority
	item1 := pq.Pop()
	if item1 == nil || item1.Request.ModelName != "high-prio" {
		t.Fatalf("expected high-prio, got %v", item1)
	}

	// Pop 2: Med priority
	item2 := pq.Pop()
	if item2 == nil || item2.Request.ModelName != "med-prio" {
		t.Fatalf("expected med-prio, got %v", item2)
	}

	// Pop 3: Low priority
	item3 := pq.Pop()
	if item3 == nil || item3.Request.ModelName != "low-prio" {
		t.Fatalf("expected low-prio, got %v", item3)
	}

	if pq.Len() != 0 {
		t.Fatalf("expected empty queue, got len %d", pq.Len())
	}
}

func TestPriorityQueueSamePriorityFIFO(t *testing.T) {
	pq := NewPriorityQueue()

	t0 := time.Now()
	pq.Push(&QueueItem{
		Request:      &pb.InferenceRequest{ModelName: "first"},
		Priority:     5,
		EnqueuedAt:   t0,
		ResponseChan: make(chan QueuedResponse, 1),
	})

	pq.Push(&QueueItem{
		Request:      &pb.InferenceRequest{ModelName: "second"},
		Priority:     5,
		EnqueuedAt:   t0.Add(5 * time.Millisecond),
		ResponseChan: make(chan QueuedResponse, 1),
	})

	item1 := pq.Pop()
	if item1.Request.ModelName != "first" {
		t.Fatalf("expected 'first', got %s", item1.Request.ModelName)
	}

	item2 := pq.Pop()
	if item2.Request.ModelName != "second" {
		t.Fatalf("expected 'second', got %s", item2.Request.ModelName)
	}
}

func TestPriorityQueueDeadlineEviction(t *testing.T) {
	pq := NewPriorityQueue()

	past := time.Now().Add(-10 * time.Millisecond)
	future := time.Now().Add(10 * time.Second)

	chExpired := make(chan QueuedResponse, 1)
	pq.Push(&QueueItem{
		Request:      &pb.InferenceRequest{ModelName: "expired"},
		Priority:     10,
		EnqueuedAt:   past,
		Deadline:     past,
		ResponseChan: chExpired,
	})

	chValid := make(chan QueuedResponse, 1)
	pq.Push(&QueueItem{
		Request:      &pb.InferenceRequest{ModelName: "valid"},
		Priority:     5,
		EnqueuedAt:   time.Now(),
		Deadline:     future,
		ResponseChan: chValid,
	})

	evicted := pq.EvictExpired(time.Now(), errors.New("timeout"))
	if evicted != 1 {
		t.Fatalf("expected 1 evicted, got %d", evicted)
	}

	// Verify channel received error
	res := <-chExpired
	if res.Err == nil {
		t.Fatal("expected error on expired channel, got nil")
	}

	// Verify valid item remains
	if pq.Len() != 1 {
		t.Fatalf("expected 1 item remaining, got %d", pq.Len())
	}
	remaining := pq.Pop()
	if remaining.Request.ModelName != "valid" {
		t.Fatalf("expected 'valid' remaining, got %s", remaining.Request.ModelName)
	}
}

func TestPriorityQueueConcurrentAccess(t *testing.T) {
	pq := NewPriorityQueue()
	var wg sync.WaitGroup

	numProducers := 20
	itemsPerProducer := 50

	// Concurrent pushes
	for i := 0; i < numProducers; i++ {
		wg.Add(1)
		go func(pid int) {
			defer wg.Done()
			for j := 0; j < itemsPerProducer; j++ {
				pq.Push(&QueueItem{
					Request:      &pb.InferenceRequest{ModelName: "concurrent"},
					Priority:     int32(j % 5),
					EnqueuedAt:   time.Now(),
					ResponseChan: make(chan QueuedResponse, 1),
				})
			}
		}(i)
	}
	wg.Wait()

	expectedTotal := numProducers * itemsPerProducer
	if pq.Len() != expectedTotal {
		t.Fatalf("expected %d items, got %d", expectedTotal, pq.Len())
	}

	// Concurrent pops
	poppedCount := 0
	var popMu sync.Mutex

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				item := pq.Pop()
				if item == nil {
					return
				}
				popMu.Lock()
				poppedCount++
				popMu.Unlock()
			}
		}()
	}
	wg.Wait()

	if poppedCount != expectedTotal {
		t.Fatalf("expected %d popped items, got %d", expectedTotal, poppedCount)
	}
}

