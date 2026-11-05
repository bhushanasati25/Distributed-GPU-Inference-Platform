variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "gpu-inference-project"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "gpu-inference-cluster"
}

