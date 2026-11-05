output "cluster_name" {
  description = "GKE Cluster Name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE Master Endpoint"
  value       = google_container_cluster.primary.endpoint
}

output "get_credentials_command" {
  description = "gcloud CLI command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id}"
}

