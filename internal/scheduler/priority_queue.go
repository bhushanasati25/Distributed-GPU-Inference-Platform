package scheduler

import (
	"container/heap"
	"sync"
	"time"

	pb "github.com/distributed-gpu-inference/platform/pkg/proto/inference"
)

// QueuedResponse returns execution results or dispatch error back to the caller channel.
type QueuedResponse struct {
	Response *pb.InferenceResponse
	Err      error
}

// QueueItem represents a pending inference request awaiting GPU execution.
type QueueItem struct {
	Request      *pb.InferenceRequest
	Priority     int32
	EnqueuedAt   time.Time
	Deadline     time.Time
	ResponseChan chan QueuedResponse
	index        int // Used internally by container/heap
}

// priorityHeap implements heap.Interface for min/max priority order.
type priorityHeap []*QueueItem

func (h priorityHeap) Len() int { return len(h) }

// Less prioritizes higher priority values first; ties broken by earliest enqueued time.
func (h priorityHeap) Less(i, j int) bool {
	if h[i].Priority != h[j].Priority {
		return h[i].Priority > h[j].Priority
	}
	return h[i].EnqueuedAt.Before(h[j].EnqueuedAt)
}

func (h priorityHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *priorityHeap) Push(x interface{}) {
	n := len(*h)
	item := x.(*QueueItem)
	item.index = n
	*h = append(*h, item)
}

func (h *priorityHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*h = old[0 : n-1]
	return item
}

// PriorityQueue provides thread-safe access to a priority-ordered queue.
type PriorityQueue struct {
	mu    sync.Mutex
	items priorityHeap
}

// NewPriorityQueue creates an empty priority queue.
func NewPriorityQueue() *PriorityQueue {
	pq := &PriorityQueue{
		items: make(priorityHeap, 0),
	}
	heap.Init(&pq.items)
	return pq
}

// Push adds an item to the queue.
func (pq *PriorityQueue) Push(item *QueueItem) {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	heap.Push(&pq.items, item)
}

// Pop removes and returns the highest priority item.
func (pq *PriorityQueue) Pop() *QueueItem {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	if len(pq.items) == 0 {
		return nil
	}
	return heap.Pop(&pq.items).(*QueueItem)
}

// Len returns current number of items.
func (pq *PriorityQueue) Len() int {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	return len(pq.items)
}

// EvictExpired removes all items whose deadline has passed and notifies their callers.
func (pq *PriorityQueue) EvictExpired(now time.Time, expiredErr error) int {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	evicted := 0
	i := 0
	for i < len(pq.items) {
		item := pq.items[i]
		if !item.Deadline.IsZero() && now.After(item.Deadline) {
			heap.Remove(&pq.items, i)
			item.ResponseChan <- QueuedResponse{Err: expiredErr}
			close(item.ResponseChan)
			evicted++
		} else {
			i++
		}
	}
	return evicted
}

