# =============================================================================
# Terraform: GCP GKE with 4 Heterogeneous GPU Accelerator Node Pools
# =============================================================================
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# GKE Cluster Control Plane
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = "default"
  subnetwork = "default"
}

# -----------------------------------------------------------------------------
# 4 GPU Accelerator Node Pools
# -----------------------------------------------------------------------------

# Pool 1: NVIDIA A100 (a2-highgpu-1g)
resource "google_container_node_pool" "pool_1_a100" {
  name     = "gpu-pool-1-a100"
  location = var.region
  cluster  = google_container_cluster.primary.name

  node_count = 1

  node_config {
    machine_type = "a2-highgpu-1g" # 1x NVIDIA A100 40GB
    guest_accelerator {
      type  = "nvidia-tesla-a100"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
    labels = {
      "node-pool"   = "gpu-pool-1"
      "gpu-node"    = "true"
      "accelerator" = "nvidia-a100"
    }
  }
}

# Pool 2: NVIDIA A100 (a2-highgpu-1g)
resource "google_container_node_pool" "pool_2_a100" {
  name     = "gpu-pool-2-a100"
  location = var.region
  cluster  = google_container_cluster.primary.name

  node_count = 1

  node_config {
    machine_type = "a2-highgpu-1g"
    guest_accelerator {
      type  = "nvidia-tesla-a100"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
    labels = {
      "node-pool"   = "gpu-pool-2"
      "gpu-node"    = "true"
      "accelerator" = "nvidia-a100"
    }
  }
}

# Pool 3: NVIDIA H100 (a3-highgpu-8g)
resource "google_container_node_pool" "pool_3_h100" {
  name     = "gpu-pool-3-h100"
  location = var.region
  cluster  = google_container_cluster.primary.name

  node_count = 1

  node_config {
    machine_type = "a3-highgpu-8g" # NVIDIA H100 80GB SXM5
    guest_accelerator {
      type  = "nvidia-h100-80gb"
      count = 8
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
    labels = {
      "node-pool"   = "gpu-pool-3"
      "gpu-node"    = "true"
      "accelerator" = "nvidia-h100"
    }
  }
}

# Pool 4: NVIDIA L4 (g2-standard-8)
resource "google_container_node_pool" "pool_4_l4" {
  name     = "gpu-pool-4-l4"
  location = var.region
  cluster  = google_container_cluster.primary.name

  node_count = 1

  node_config {
    machine_type = "g2-standard-8" # 1x NVIDIA L4 24GB
    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
    labels = {
      "node-pool"   = "gpu-pool-4"
      "gpu-node"    = "true"
      "accelerator" = "nvidia-l4"
    }
  }
}

