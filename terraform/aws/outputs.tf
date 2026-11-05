output "cluster_endpoint" {
  description = "EKS Control Plane API endpoint"
  value       = aws_eks_cluster.platform.endpoint
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = aws_eks_cluster.platform.name
}

output "kubeconfig_command" {
  description = "AWS CLI command to configure local kubectl"
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${aws_eks_cluster.platform.name}"
}

