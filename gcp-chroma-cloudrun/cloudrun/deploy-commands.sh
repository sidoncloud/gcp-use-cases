gcloud run deploy info-chatbot \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --max-instances 2 \
  --min-instances 1