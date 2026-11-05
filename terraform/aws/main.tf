# =============================================================================
# Terraform: AWS EKS with 4 Heterogeneous GPU Node Groups
# =============================================================================
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC & Networking
resource "aws_vpc" "gpu_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.cluster_name}-vpc"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

resource "aws_subnet" "gpu_subnets" {
  count                   = 2
  vpc_id                  = aws_vpc.gpu_vpc.id
  cidr_block              = cidrsubnet(aws_vpc.gpu_vpc.cidr_block, 4, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.cluster_name}-subnet-${count.index}"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

data "aws_availability_zones" "available" {}

# EKS Cluster Control Plane
resource "aws_eks_cluster" "platform" {
  name     = var.cluster_name
  version  = var.kubernetes_version
  role_arn = aws_iam_role.cluster_role.arn

  vpc_config {
    subnet_ids = aws_subnet.gpu_subnets[*].id
  }
}

# -----------------------------------------------------------------------------
# 4 GPU Node Groups
# -----------------------------------------------------------------------------

# GPU Node Group 1: NVIDIA A100 (p4d.24xlarge)
resource "aws_eks_node_group" "pool_1_a100" {
  cluster_name    = aws_eks_cluster.platform.name
  node_group_name = "gpu-pool-1-a100"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.gpu_subnets[*].id
  instance_types  = ["p4d.24xlarge"]
  ami_type        = "AL2_x86_64_GPU"

  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 4
  }

  labels = {
    "node-pool"   = "gpu-pool-1"
    "gpu-node"    = "true"
    "accelerator" = "nvidia-a100"
  }
}

# GPU Node Group 2: NVIDIA A100 (p4d.24xlarge)
resource "aws_eks_node_group" "pool_2_a100" {
  cluster_name    = aws_eks_cluster.platform.name
  node_group_name = "gpu-pool-2-a100"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.gpu_subnets[*].id
  instance_types  = ["p4d.24xlarge"]
  ami_type        = "AL2_x86_64_GPU"

  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 4
  }

  labels = {
    "node-pool"   = "gpu-pool-2"
    "gpu-node"    = "true"
    "accelerator" = "nvidia-a100"
  }
}

# GPU Node Group 3: NVIDIA H100 (p5.48xlarge)
resource "aws_eks_node_group" "pool_3_h100" {
  cluster_name    = aws_eks_cluster.platform.name
  node_group_name = "gpu-pool-3-h100"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.gpu_subnets[*].id
  instance_types  = ["p5.48xlarge"]
  ami_type        = "AL2_x86_64_GPU"

  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 2
  }

  labels = {
    "node-pool"   = "gpu-pool-3"
    "gpu-node"    = "true"
    "accelerator" = "nvidia-h100"
  }
}

# GPU Node Group 4: NVIDIA L4 / A10G (g5.12xlarge)
resource "aws_eks_node_group" "pool_4_l4" {
  cluster_name    = aws_eks_cluster.platform.name
  node_group_name = "gpu-pool-4-l4"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.gpu_subnets[*].id
  instance_types  = ["g5.12xlarge"]
  ami_type        = "AL2_x86_64_GPU"

  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 4
  }

  labels = {
    "node-pool"   = "gpu-pool-4"
    "gpu-node"    = "true"
    "accelerator" = "nvidia-l4"
  }
}

# IAM Roles
resource "aws_iam_role" "cluster_role" {
  name = "${var.cluster_name}-cluster-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "eks.amazonaws.com" } }]
  })
}

resource "aws_iam_role" "node_role" {
  name = "${var.cluster_name}-node-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" } }]
  })
}

